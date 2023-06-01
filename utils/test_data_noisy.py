#This program prepares the test data for evaluation
import sys
import os
sys.path.append(os.getcwd())
from subgraph_extraction.extract_noisy_conf import triplet_rule_paths


def put_labels(test):
    # Put labels
    test_final_with_label = {}

    for key, val in test.items():
        test_batch_with_label = []
        if len(val)>0:
            for v in val:
                v = [v, [1]]
                test_batch_with_label.append(v)
            test_final_with_label[key] = test_batch_with_label
        else:
            test_final_with_label[key] = test_batch_with_label
    return test_final_with_label

def restructure(atom):
    rel = atom[1]
    atom[1] = atom[2]
    atom[2] = rel
    return atom

def encode_test(test_data, rel2id, entity2id_train): #Same for test_ind
    encoded_test_batch = {}
    for key, test in test_data.items():
        key = key.split('\t')
        key = restructure(key)
        key[0] = entity2id_train[key[0]]
        key[1] = entity2id_train[key[1]]
        key[2] = rel2id[key[2]]
        key = str(key[0]) + '\t' + str(key[1]) + '\t' + str(key[2])

        if len(test)>0:
            for test_item in test:
                test_item = restructure(test_item)
                test_item[0] = entity2id_train[test_item[0]]
                test_item[1] = entity2id_train[test_item[1]]
                test_item[2] = rel2id[test_item[2]]
            encoded_test_batch[key] = test
        else:
            encoded_test_batch[key] = test
    return encoded_test_batch


def get_rule_paths_test_ind(test_data, rel2id, entity2id_train, num_ins, rules, h2r2t, workers):
    test = encode_test(test_data, rel2id, entity2id_train)
    test = put_labels(test)
    rule_path_for_test = {}
    for key, test_batch in test.items():
        rule_paths_test_batch = triplet_rule_paths(test_batch, rules, h2r2t, num_ins, workers)
        rule_path_for_test[key] = rule_paths_test_batch
    return rule_path_for_test

