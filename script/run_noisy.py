# Train, Validate, and Evaluate over Test Data
import os
import sys
sys.path.append(os.getcwd())
from utils.encode_triplets import encode_train_ind
import random
import time
import pickle
import re
import argparse
import collections
import torch
import math
import operator


def shuffle_batch(batch_score):
    shuffled_batch = {}
    keys = list(batch_score.keys())
    random.shuffle(keys)
    for key in keys:
        shuffled_batch[key] = batch_score[key]
    return shuffled_batch

def eval_trad_hit(ranking, threshold):
    score = 0
    for rank in ranking:
        if int(rank) < threshold+1:
            score+=1
    return score/len(ranking)

def eval_approximated_hit(ranking, threshold, num_samples, num_negatives):
    ranking = torch.tensor(ranking)
    num_negatives = torch.tensor(num_negatives)
    threshold = int(threshold)
    num_samples = int(num_samples)
    # Compute Hits@10_50
    false_pos_rate = (ranking - 1).float() / num_negatives
    score = 0
    for i in range(threshold):
        combs = math.factorial(num_samples - 1) / math.factorial(i) / math.factorial(num_samples - i - 1)
        score += combs * (false_pos_rate ** i) * ((1 - false_pos_rate) ** (num_samples - i - 1))
    score = score.mean()
    return score

def eval_mrr(ranking):
    ranking = torch.tensor(ranking)
    score = (1 / ranking.float()).mean()
    return score

def get_metric_score(ranking_dict, metric, num_negatives):
    print()
    print()
    for rank_type, ranking in ranking_dict.items():
        for met in metric:
            if met == 'mrr':
                score = eval_mrr(ranking)
                print('MRR for', rank_type, ':', score)
            elif '_' in met:
                threshold = met.split('@')[1].split('_')[0]
                num_samples = met.split('@')[1].split('_')[1]
                score = eval_approximated_hit(ranking, threshold, num_samples, num_negatives)
                print('Hits@10_50 for', rank_type, ':', score)
            else:
                threshold = int(met.split('@')[1])
                score = eval_trad_hit(ranking, threshold)
                print('Hits@%d for ' %threshold, rank_type, ':',  score)
        print('----------------------------------------------------------------------------')


def get_hybrid_rank(test_strict_ind, nbfnet_score, sorted_batch_score, batch_key):
    anyburl_indices = []
    for key in sorted_batch_score:
        anyburl_indices.append(int(key.split('\t')[1]))

    batch_score = {}
    for index in test_strict_ind:
        key = batch_key.split('\t')[0] + '\t' + str(index) + '\t' + batch_key.split('\t')[2]
        if index in anyburl_indices:
            scr = sorted_batch_score[key] + 10000 # To keep AnyBURL predictions at top
            batch_score[key] = float(scr)
        else:
            batch_score[key] = float(nbfnet_score[index])

    sorted_batch_score_final = dict(sorted(batch_score.items(), key=operator.itemgetter(1), reverse=True))
    batch_key_pred_pos = list(sorted_batch_score_final).index(batch_key)
    rank = batch_key_pred_pos+1

    return rank


def test_noisy(data_dir, test_scores, num_negatives, nbfnet_rank_dict, num_rel, test_strict_ind,
               nbfnet_score_dict, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    metric = ['mrr', 'hits@1', 'hits@3', 'hits@10', 'hits@10_50']

    ranking_dict = collections.defaultdict(list)
    # Different Rankings to fetch
    ranking_original = []  # Only the Noisy indices

    ranking_random = []  # Noisy indices + random rank for non-Noisy

    ranking_original_nbfnet_0 = []  # Noisy + NBFNet run 1
    ranking_original_nbfnet_1 = []  # Noisy + NBFNet run 2
    ranking_original_nbfnet_2 = []  # Noisy + NBFNet run 3
    ranking_original_nbfnet_3 = []  # Noisy + NBFNet run 4
    ranking_original_nbfnet_4 = []  # Noisy + NBFNet run 5
    '''
    ranking_ab_ind_nbfnet_0 = []  # NBFNet + NBFNet run 1
    ranking_ab_ind_nbfnet_1 = []  # NBFNet + NBFNet run 2
    ranking_ab_ind_nbfnet_2 = []  # NBFNet + NBFNet run 3
    ranking_ab_ind_nbfnet_3 = []  # NBFNet + NBFNet run 4
    ranking_ab_ind_nbfnet_4 = []  # NBFNet + NBFNet run 5
    '''
    batch_no = -1
    test_scores_final = []
    for batch_key, batch_triplets in test_scores.items():
        #print('batch key', batch_key)
        #print('batch triplets', batch_triplets)

        batch_no += 1
        sample_keys = []
        sample_scores = []
        for sample_key, sample_score in batch_triplets.items():
            sample_keys.append(sample_key.split('\t')[:-1])
            sample_scores.append(sample_score)
        if len(sample_scores) > 0:
            scores = sample_scores
            batch_score = {}
            for i in range(len(sample_keys)):
                s_key = str(sample_keys[i][0]) + '\t' + str(sample_keys[i][1]) + '\t' + str(sample_keys[i][2])
                batch_score[s_key] = scores[i]
            # Store the batch_scores
            test_scores_final.append(batch_score)
            # Suffle the batch_score to break the ties randomly
            batch_score_shuffled = shuffle_batch(batch_score)
            #print('batch score shuffled', batch_score_shuffled)
            sorted_batch_score = dict(sorted(batch_score_shuffled.items(), key=operator.itemgetter(1), reverse=True))
            #print('batch score sorted', sorted_batch_score)
            #print('===========================================================')
            if batch_key in sorted_batch_score:  # Case 1: The Pos has a rule
                batch_key_pred_pos = list(sorted_batch_score).index(batch_key)
                ranking_original.append(int(batch_key_pred_pos) + 1)
                ranking_random.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_0.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_1.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_2.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_3.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_4.append(int(batch_key_pred_pos) + 1)

                '''
                # Get ranking of nbfnet on anyburl index at top  for nbfnet + nbfnet
                # print('Working for Case')
                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[0][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_0.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[1][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_1.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[2][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_2.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[3][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_3.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[4][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_4.append(rank_ab_ind_nbfnet)
                '''
            else:  # Case 3: Some Negs have rule but Pos does not
                ranking_original.append(num_negatives[batch_no])  # Assigning the least score if no rules
                rank = random.randint(len(sample_scores) + 1, num_negatives[batch_no] + 1)
                ranking_random.append(rank)
                # Get the hybrid rank, i.e., assign scores to sample triplets from AnyBURL_gnn
                # if they have a rule, else get the score from NBFNet for other indices
                # in the test_strict_indices, and then find the rank.

                rank_hybrid = get_hybrid_rank(test_strict_ind[batch_no], nbfnet_score_dict[0][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_original_nbfnet_0.append(rank_hybrid)

                rank_hybrid = get_hybrid_rank(test_strict_ind[batch_no], nbfnet_score_dict[1][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_original_nbfnet_1.append(rank_hybrid)

                rank_hybrid = get_hybrid_rank(test_strict_ind[batch_no], nbfnet_score_dict[2][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_original_nbfnet_2.append(rank_hybrid)

                rank_hybrid = get_hybrid_rank(test_strict_ind[batch_no], nbfnet_score_dict[3][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_original_nbfnet_3.append(rank_hybrid)

                rank_hybrid = get_hybrid_rank(test_strict_ind[batch_no], nbfnet_score_dict[4][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_original_nbfnet_4.append(rank_hybrid)

                '''
                # Get ranking of nbfnet on anyburl index at top + nbfnet
                # print('Working for Case 3')
                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[0][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_0.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[1][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_1.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[2][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_2.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[3][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_3.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[4][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_4.append(rank_ab_ind_nbfnet)
                '''

        else:  # Case 2: No rules for Pos as well as Negs.
            ranking_original.append(num_negatives[batch_no])  # Assigning the least score if no rules

            rank = random.randint(0, num_negatives[batch_no] + 1)
            ranking_random.append(rank)

            ranking_original_nbfnet_0.append(int(nbfnet_rank_dict[0][batch_no]))
            ranking_original_nbfnet_1.append(int(nbfnet_rank_dict[1][batch_no]))
            ranking_original_nbfnet_2.append(int(nbfnet_rank_dict[2][batch_no]))
            ranking_original_nbfnet_3.append(int(nbfnet_rank_dict[3][batch_no]))
            ranking_original_nbfnet_4.append(int(nbfnet_rank_dict[4][batch_no]))

            '''
            ranking_ab_ind_nbfnet_0.append(int(nbfnet_rank_dict[0][batch_no]))
            ranking_ab_ind_nbfnet_1.append(int(nbfnet_rank_dict[1][batch_no]))
            ranking_ab_ind_nbfnet_2.append(int(nbfnet_rank_dict[2][batch_no]))
            ranking_ab_ind_nbfnet_3.append(int(nbfnet_rank_dict[3][batch_no]))
            ranking_ab_ind_nbfnet_4.append(int(nbfnet_rank_dict[4][batch_no]))
            '''
    ranking_dict['Noisy'] = ranking_random
    ranking_dict['Noisy_NBFNet1'] = ranking_original_nbfnet_0
    ranking_dict['Noisy_NBFNet2'] = ranking_original_nbfnet_1
    ranking_dict['Noisy_NBFNet3'] = ranking_original_nbfnet_2
    ranking_dict['Noisy_NBFNet4'] = ranking_original_nbfnet_3
    ranking_dict['Noisy_NBFNet5'] = ranking_original_nbfnet_4

    '''
    ranking_dict['NBFNet_NBFNet1'] = ranking_ab_ind_nbfnet_0
    ranking_dict['NBFNet_NBFNet2'] = ranking_ab_ind_nbfnet_1
    ranking_dict['NBFNet_NBFNet3'] = ranking_ab_ind_nbfnet_2
    ranking_dict['NBFNet_NBFNet4'] = ranking_ab_ind_nbfnet_3
    ranking_dict['NBFNet_NBFNet5'] = ranking_ab_ind_nbfnet_4
    '''
    print('============================================================================')
    print('Noisy_Random', ranking_random)
    print('============================================================================')
    print('Noisy_NBFNet1', ranking_original_nbfnet_0)
    print('============================================================================')
    #print('NBFNet_NBFNet1', ranking_ab_ind_nbfnet_0)
    print('============================================================================')

    get_metric_score(ranking_dict, metric, num_negatives)


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
    parser.add_argument('-nnbf', '--num_nbfnet_runs', help='Number of NBFNet runs', default=5)
    parser.add_argument('-n', '--num_ins', help='No of Instantiations', default=0)
    parser.add_argument('-out', '--out', help='Out file with noisy test scores saved',
                        default='test_all_rule')

    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    data_dir = args.in_dir
    num_nbfnet_runs = int(args.num_nbfnet_runs)
    num_ins = int(args.num_ins)
    out = args.out

    print('Arguments', args)

    #path_dir = os.path.join(data_dir, out)
    # Read the scores Test data
    test_scores = pickle.load(open(os.path.join(data_dir, 'noisy_rank', out + '.pkl'), 'rb'))

    train_ind_file = open(os.path.join(data_dir, 'train_ind.txt'))
    # Read and Process the NBFNet ranks
    nbfnet_rank_dict = process_nbfnet_rank(data_dir, num_nbfnet_runs)
    # Read and Process the NBFNet score
    nbfnet_score_dict = process_nbfnet_score(data_dir, num_nbfnet_runs, device)

    # Extracting strict negatives triplets for the given test triplet
    # This is to ensure that no existing test triplets in train_ind should be considered as negative test_ind
    test_strict_ind, num_negatives, rel2id = get_num_neg(data_dir, train_ind_file, test_scores)
    num_rel = len(rel2id)

    test_noisy(data_dir, test_scores, num_negatives, nbfnet_rank_dict, num_rel, test_strict_ind,
               nbfnet_score_dict, args)


    end = time.time()
    print('Time taken in Training and Testing :  ', (end - start)/60, ' Minutes.')





