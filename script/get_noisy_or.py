# This program evalautes noisy-or confidences for the test data

# Imports...
import sys
import os
sys.path.append(os.getcwd())
from utils.encode_triplets import encode, entity2rel2entity, put_labels, encode_valid_pos_neg, encode_train_ind
from utils.encode_rules_noisy import enc
from utils.neg_triplet import extract_neg
#from subgraph_extraction.extract_5_rule_paths import triplet_rule_paths # Choose only five rule-paths
from subgraph_extraction.extract_rule_paths import triplet_rule_paths # Choose atmost five rules and their path
from utils.generate_triplets_to_test import generate_batch_triplets
from utils.test_data_noisy import get_rule_paths_test_ind
import time
import pickle
import argparse



if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument('-d', '--data', help='Data Directory', default='../data/nell_v3')
    #parser.add_argument('-r', '--rule', help='Rule Directory', default='../anyburl-22/nell_v3')
    parser.add_argument('-d', '--data', help='Data Directory', default='../data/fb15k237_v1')
    parser.add_argument('-r', '--rule', help='Rule Directory', default='../anyburl-22/fb15k237_v1')
    parser.add_argument('-n', '--num_ins', help='No of Instantiations', default=0)  # 5 and 1000
    parser.add_argument('-remove_hops', '--remove_hops', help='Remove Rules till the given hops', default=0)
    parser.add_argument('-w', '--workers', help='No of workers', default=7)
    parser.add_argument('-o', '--out', help='out_file', default='test_all_rule')

    args = parser.parse_args()
    data_dir = args.data
    rule_dir = args.rule
    num_ins = int(args.num_ins)
    remove_hops = args.remove_hops
    workers = int(args.workers)
    out = args.out

    rule_file = open(os.path.join(rule_dir, 'rules'))
    rules = enc(rule_file, data_dir, remove_hops)  # Rules are sorted and encoded with reverse triplets with conf

    # Reading fact graph for test
    train_ind_file = open(os.path.join(data_dir, 'train_ind.txt'))
    train_ind, entity2id, rel2id = encode_train_ind(data_dir, train_ind_file)
    h2r2t = entity2rel2entity(train_ind)
    #out_h2r2t = open(os.path.join(data_dir, 'test_h2r2t.pkl'), 'wb')

    test_batch_triplets = generate_batch_triplets(rule_dir, 'test_preds')

    #print('test_batch_triplets', test_batch_triplets)
    if num_ins == 0:
        num_ins = len(rules)
    rule_conf_test = get_rule_paths_test_ind(test_batch_triplets, rel2id, entity2id, num_ins, rules, h2r2t, workers)
    print('Writing the test scores')
    out_file = open(os.path.join(data_dir, 'noisy_rank', out +'.pkl'), 'wb')
    pickle.dump(rule_conf_test, out_file)

    end = time.time()
    print('Total time taken in extracting rule path instantiations: ', end - start, ' Secs.')