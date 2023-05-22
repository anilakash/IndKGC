# This program will extract rule paths for train, valid, and test_ind batches.
import sys
import os
sys.path.append(os.getcwd())
from utils.encode_triplets import encode, entity2rel2entity, put_labels, encode_valid_pos_neg, encode_train_ind
from utils.encode_rules import enc
from utils.neg_triplet import extract_neg
from subgraph_extraction.extract_rule_paths import triplet_rule_paths
from utils.generate_triplets_to_test import generate_batch_triplets
from utils.test_data import get_rule_paths_test_ind
import time
import pickle
import argparse


def rule_paths_train(data_dir, train, rules, h2r2t, num_ins, workers, out_path, num_con):
    path_dir = os.path.join(data_dir, 'top_%d' % num_ins + out_path)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    if os.path.exists(os.path.join(path_dir, 'train.pkl')):
        print('Rule path exist for Train.')
    else:
        print('Extracting rule path instantiation for train, it may take a while...')
        rule_paths_train = triplet_rule_paths(train, rules, h2r2t, num_ins, workers, num_con)
        print('Writing the rule paths for train')
        out_file = open(os.path.join(path_dir, 'train.pkl'), 'wb')
        pickle.dump(rule_paths_train, out_file)

def rule_paths_valid(data_dir, valid, rules, h2r2t, num_ins, workers, out_path, num_con):
    path_dir = os.path.join(data_dir, 'top_%d' % num_ins + out_path)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    if os.path.exists(os.path.join(path_dir, 'valid.pkl')):
        print('Rule path exist for Valid.')
    else:
        print('Extracting rule path instantiation for Valid, it may take a while...')
        rule_paths_valid = triplet_rule_paths(valid, rules, h2r2t, num_ins, workers, num_con)
        print('Writing the rule paths for Valid')
        out_file = open(os.path.join(path_dir, 'valid.pkl'), 'wb')
        pickle.dump(rule_paths_valid, out_file)

def rule_paths_test(data_dir, test_batch_triplets, rel2id, entity2id, num_ins, rules, h2r2t, workers, out_path, num_con):
    path_dir = os.path.join(data_dir, 'top_%d' % num_ins + out_path)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    if os.path.exists(os.path.join(path_dir, 'test.pkl')):
        print('Rule path exist for Test.')
    else:
        print('Extracting rule path instantiation for Test, it may take a while...')
        rule_paths_test = get_rule_paths_test_ind(test_batch_triplets, rel2id, entity2id, num_ins,
                                                  rules, h2r2t, workers, num_con)
        print('Writing the rule paths for Test')
        out_file = open(os.path.join(path_dir, 'test.pkl'), 'wb')
        pickle.dump(rule_paths_test, out_file)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data Directory', default='../data/fb15k237_v1')
    parser.add_argument('-r', '--rule', help='Rule Directory', default='../anyburl-22/fb15k237_v1')
    parser.add_argument('-n', '--num_ins', help='No of Instantiations', default=5)   # 5 and 1000
    parser.add_argument('-nn', '--num_neg', help='No of negative triplets per triplet', default=10)  # Fix to 10
    parser.add_argument('-w', '--workers', help='No of workers', default=7)
    parser.add_argument('-o', '--out', help='Out path to save output', default='_paths_context_4_hop_1_rem_hop1') # Need Perm
    parser.add_argument('-nc', '--num_con', help='No of Context', default=0)  # Need perm
    parser.add_argument('-remove_hops', '--remove_hops', help='Remove Rules till the given hops', default=1)

    args = parser.parse_args()
    data_dir = args.data
    rule_dir = args.rule
    num_ins = int(args.num_ins)
    num_neg = int(args.num_neg)
    workers = int(args.workers)
    out_path = args.out
    num_con = int(args.num_con)
    remove_hops = args.remove_hops

    rule_file = open(os.path.join(rule_dir, 'rules'))
    train_pos_file = open(os.path.join(data_dir, 'train.txt'))
    train_neg = extract_neg(rule_dir, num_neg, 'train_preds')

    # Generate training data
    train_pos, train_neg, entity2id, rel2id = encode(data_dir, train_pos_file, train_neg)
    # Generate Validation data
    val_pos_file = open(os.path.join(data_dir, 'valid.txt'))
    val_neg = extract_neg(rule_dir, num_neg, 'val_preds')
    val_pos, val_neg = encode_valid_pos_neg(val_pos_file, val_neg, entity2id, rel2id)
    h2r2t = entity2rel2entity(train_pos)
    rules = enc(rule_file, data_dir, remove_hops)  # Rules are sorted and encoded with reverse triplets

    train_pos, train_neg = put_labels(train_pos, train_neg)
    train = train_pos + train_neg
    val_pos, val_neg = put_labels(val_pos, val_neg)
    valid = val_pos + val_neg

    rule_paths_train(data_dir, train, rules, h2r2t, num_ins, workers, out_path, num_con)
    rule_paths_valid(data_dir, valid, rules, h2r2t, num_ins, workers, out_path, num_con)
    # Processing Test data
    # Reading fact graph for test
    train_ind_file = open(os.path.join(data_dir, 'train_ind.txt'))
    train_ind, entity2id, rel2id = encode_train_ind(data_dir, train_ind_file)
    h2r2t = entity2rel2entity(train_ind)
    test_batch_triplets = generate_batch_triplets(rule_dir, 'test_preds')
    rule_paths_test(data_dir, test_batch_triplets, rel2id, entity2id, num_ins, rules, h2r2t, workers,
                    out_path, num_con)

    end = time.time()
    print('Total time taken in extracting rule path instantiations: ', end - start, ' Secs.')


