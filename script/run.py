# Train, Validate, and Evaluate over Test Data
import os
import sys
sys.path.append(os.getcwd())
from utils.encode_triplets import encode_train_ind
from utils.dataset_train_compgcn import create_train_graph
from utils.dataset_valid_compgcn import create_valid_graph
from utils.dataset_test_compgcn import create_test_graph
from train.train import train_test
import time
import pickle
import re
import argparse
import collections
import torch


def process_nbfnet_rank(data_dir, num_nbfnet_runs):
    nbfnet_rank_dict = collections.defaultdict(list)
    for i in range(num_nbfnet_runs):
        ranks = open(data_dir + '/nbfnet_ranks/' + str(i + 1) + '.txt').read()
        ranks = re.sub('\[', '', ranks)
        ranks = re.sub('\]', '', ranks)
        ranks = ranks.split(',')
        ranks = [item.strip() for item in ranks]
        nbfnet_rank_dict[i] = ranks
    return nbfnet_rank_dict

def process_nbfnet_score(data_dir, num_nbfnet_runs, device):
    nbfnet_score_dict = collections.defaultdict(list)
    for i in range(num_nbfnet_runs):
        score = torch.load(data_dir+'/nbfnet_ranks/nbfnet_scores_' + str(i+1) + '.pt', map_location=device)
        triplet_scores = []
        for j in range(len(score)//2):
            tail_score = score[j*2]
            head_score = score[j*2+1]
            for k in range(len(tail_score)):
                triplet_scores.append(tail_score[k])
                triplet_scores.append(head_score[k])
        nbfnet_score_dict[i] = triplet_scores
    return nbfnet_score_dict

def get_num_neg(data_dir, train_ind_file, rule_graph_for_test):
    train_ind, entity2id, rel2id = encode_train_ind(data_dir, train_ind_file)
    train_ind = [tuple(item) for item in train_ind]
    train_ind = set(train_ind)
    # print(type(train_ind))
    test_ind = []
    for key in rule_graph_for_test:
        test_ind.append((int(key.split('\t')[0]), int(key.split('\t')[1]), int(key.split('\t')[2])))
    num_negatives = []
    test_strict_ind = []
    for test in test_ind:
        test_batch = []
        for i in range(len(entity2id)):
            triplet = (test[0], i, test[2])
            test_batch.append(triplet)
        test_batch = set(test_batch)
        intersect = test_batch.intersection(train_ind)
        test_sub = test_batch.difference(train_ind)

        # Find valid test indices
        ind = []
        for test in test_sub:
            ind.append(test[1])
        test_strict_ind.append(sorted(ind))
        # Calculate number of negatives for each test candidate
        num_neg = len(test_batch) - len(intersect) - 1  # total possible test - available in train_ind - candidate
        num_negatives.append(num_neg)
    return test_strict_ind, num_negatives, rel2id


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--in_dir', help='Input Directory', default='../data/fb15k237_v1')
    parser.add_argument('-m', '--model_run', help='Iteration num of Model', default='run_1_model')
    parser.add_argument('-nnbf', '--num_nbfnet_runs', help='Number of NBFNet runs', default=5)
    parser.add_argument('-n', '--num_ins', help='No of Instantiations', default=5)
    parser.add_argument('-l', '--len_rule', help='Max rule length or no of atoms in the body', default=4)
    parser.add_argument('-nh', '--num_hidden', help='Dimension of hidden layers', default=64)
    parser.add_argument('-lr', '--learning_rate', help='Initial Learning Rate', default=0.004)
    parser.add_argument('-b', '--num_bases', help='Number of bases', default=-1)
    parser.add_argument('-do', '--drop_out', help='Dropout probability', default=0.1)
    parser.add_argument('-e', '--epoch', help='Max epochs', default=50)
    parser.add_argument('-p', '--patience', help='Patience value for early stop', default=20)
    parser.add_argument('-o', '--opn', help='Compositional Operator', default='sub')
    parser.add_argument('-bs', '--bias', help='Bias', default=False)
    parser.add_argument('-a', '--act', help='Activation', default=torch.relu)
    parser.add_argument('-c1', '--concat1', help='Concatenate Graph Emb with Rel Emb', default=False) #2
    parser.add_argument('-c2', '--concat2', help='Concatenate Graph Emb, Rel Emb, Head, Tail', default=False) #4
    parser.add_argument('-c3', '--concat3', help='Concatenate Graph Emb with Projection of |Tail-Head| on Rel Emb ', default=False)
    parser.add_argument('-c4', '--concat4', help='Concatenate Graph Emb with Projection of |Head * Tail| on Rel Emb ', default=False)
    parser.add_argument('-proj1', '--projection1', help='Projection of Graph Embedding on Rel Emb ', default=False) # 1
    parser.add_argument('-proj2', '--projection2', help='Projection of Graph Embedding on |Rel Emb - |Head - Tail| |', default=False)

    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    data_dir = args.in_dir
    num_nbfnet_runs = int(args.num_nbfnet_runs)
    num_ins = int(args.num_ins)
    max_rule_length = int(args.len_rule)

    print('Arguments', args)
    path_dir = os.path.join(data_dir, 'top_%d' % num_ins + '_paths')
    # Read the path instantiations for Train, Valid, and Test data
    rule_paths_train = pickle.load(open(path_dir + '/train.pkl', 'rb'))
    rule_paths_valid = pickle.load(open(path_dir + '/valid.pkl', 'rb'))
    rule_paths_test = pickle.load(open(path_dir + '/test.pkl', 'rb'))

    train_ind_file = open(os.path.join(data_dir, 'train_ind.txt'))
    # Read and Process the NBFNet ranks
    nbfnet_rank_dict = process_nbfnet_rank(data_dir, num_nbfnet_runs)
    # Read and Process the NBFNet score
    nbfnet_score_dict = process_nbfnet_score(data_dir, num_nbfnet_runs, device)

    # Extracting strict negatives triplets for the given test triplet
    # This is to ensure that no existing test triplets in train_ind should be considered as negative test_ind
    test_strict_ind, num_negatives, rel2id = get_num_neg(data_dir, train_ind_file, rule_paths_test)

    # Create graphs for train, valid, and test_ind triplets
    train_graph = create_train_graph(rule_paths_train, max_rule_length)
    valid_graph = create_valid_graph(rule_paths_valid, max_rule_length)
    test_graph = create_test_graph(rule_paths_test, max_rule_length)

    # Train and test
    num_rel = len(rel2id)
    train_test(data_dir, train_graph, valid_graph, test_graph, num_negatives, nbfnet_rank_dict,
               num_rel, test_strict_ind, nbfnet_score_dict, args)

    end = time.time()
    print('Time taken in Training and Testing :  ', (end - start)/60, ' Minutes.')





