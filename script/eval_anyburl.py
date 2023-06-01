# This program evaluates AnyBURL predictions in three setups, i.e., AnyBURL_Original,
# AnyBURL + Random, and AnyBURL + NBFNet.
import sys
import os
import torch
sys.path.append(os.getcwd())
import random
from utils.encode_triplets import encode_train_ind
from utils.gen_anyburl_batch_triplets_with_scores import generate_batch_triplets_with_scores
import operator
import collections
import re
import math
import argparse


def shuffle_batch(batch_score):
    shuffled_batch = {}
    keys = list(batch_score.keys())
    #print(keys)
    random.shuffle(keys)
    #print(keys)
    for key in keys:
        shuffled_batch[key] = batch_score[key]
    return shuffled_batch


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

def process_nbfnet_score(data_dir, num_nbfnet_runs):
    nbfnet_score_dict = collections.defaultdict(list)
    for i in range(num_nbfnet_runs):
        score = torch.load(data_dir+'/nbfnet_ranks/nbfnet_scores_' + str(i+1) + '.pt'
                           , map_location=torch.device('cpu'))
        triplet_scores = []
        for j in range(len(score)//2):
            tail_score = score[j*2]
            head_score = score[j*2+1]
            for k in range(len(tail_score)):
                triplet_scores.append(tail_score[k])
                triplet_scores.append(head_score[k])
        nbfnet_score_dict[i] = triplet_scores
    return nbfnet_score_dict

def get_num_neg(train_ind, test_batch_triplet_scores, entity2id, rel2id):
    train_ind = [tuple(item) for item in train_ind]
    train_ind = set(train_ind)
    test_ind = []
    for key in test_batch_triplet_scores:
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


def get_hybrid_rank(test_strict_ind, nbfnet_score, sorted_batch_score, batch_key):
    anyburl_indices = []
    for key in sorted_batch_score:
        anyburl_indices.append(int(key.split('\t')[1]))

    batch_score = {}
    for index in test_strict_ind:
        key = batch_key.split('\t')[0] + '\t' + str(index) + '\t' + batch_key.split('\t')[2]
        if index in anyburl_indices:
            scr = float(sorted_batch_score[key]) + 10000 # To keep AnyBURL predictions at top
            batch_score[key] = float(scr)
        else:
            batch_score[key] = float(nbfnet_score[index])

    sorted_batch_score_final = dict(sorted(batch_score.items(), key=operator.itemgetter(1), reverse=True))
    batch_key_pred_pos = list(sorted_batch_score_final).index(batch_key)
    rank = batch_key_pred_pos+1

    return rank

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


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Data Directory', default='../data/fb15k237_v1')
    parser.add_argument('-r', '--rule', help='Rule Directory', default='../anyburl-22/fb15k237_v1')
    parser.add_argument('-nnbf', '--num_nbfnet_runs', help='Number of NBFNet runs', default=5)

    args = parser.parse_args()
    data_dir = args.data_dir
    pred_test_dir = args.rule
    num_nbfnet_runs = int(args.num_nbfnet_runs)

    train_ind_file = open(os.path.join(data_dir, 'train_ind.txt'))
    train_ind, entity2id, rel2id = encode_train_ind(data_dir, train_ind_file)
    test_batch_triplet_scores = generate_batch_triplets_with_scores(pred_test_dir, 'test_preds', entity2id, rel2id)
    # Read and Process the NBFNet ranks
    nbfnet_rank_dict = process_nbfnet_rank(data_dir, num_nbfnet_runs)
    # Read and Process the NBFNet score
    nbfnet_score_dict = process_nbfnet_score(data_dir, num_nbfnet_runs)
    # Extracting strict negatives triplets for the given test triplet
    # This is to ensure that no existing test triplets in train_ind should be considered as negative test_ind
    test_strict_ind, num_negatives, rel2id = get_num_neg(train_ind, test_batch_triplet_scores, entity2id,
                                                         rel2id)
    ranking_dict = collections.defaultdict(list)
    metric = ['mrr', 'hits@1', 'hits@3', 'hits@10', 'hits@10_50']
    # Different Rankings to fetch
    ranking_original = []  # Only the AnyBURL indices

    ranking_random = []  # AnyBURL  + random rank for non-AnyBURL

    ranking_original_nbfnet_0 = []  # AnyBURL + NBFNet run 1
    ranking_original_nbfnet_1 = []  # AnyBURL + NBFNet run 2
    ranking_original_nbfnet_2 = []  # AnyBURL + NBFNet run 3
    ranking_original_nbfnet_3 = []  # AnyBURL + NBFNet run 4
    ranking_original_nbfnet_4 = []  # AnyBURL + NBFNet run 5

    batch_no = -1

    for batch_key, batch_score in test_batch_triplet_scores.items():
        batch_no += 1
        if len(batch_score)>0:
            batch_score_shuffled = shuffle_batch(batch_score)
            sorted_batch_score = dict(sorted(batch_score_shuffled.items(), key=operator.itemgetter(1), reverse=True))
            if batch_key in sorted_batch_score:  # Case 1: The Pos has a rule
                batch_key_pred_pos = list(sorted_batch_score).index(batch_key)
                ranking_original.append(int(batch_key_pred_pos) + 1)
                ranking_random.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_0.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_1.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_2.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_3.append(int(batch_key_pred_pos) + 1)
                ranking_original_nbfnet_4.append(int(batch_key_pred_pos) + 1)
            else:
                ranking_original.append(num_negatives[batch_no])  # Assigning the least score if no rules
                rank = random.randint(len(batch_score) + 1, num_negatives[batch_no] + 1)
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

        else:
            ranking_original.append(num_negatives[batch_no])  # Assigning the least score if no rules

            rank = random.randint(0, num_negatives[batch_no] + 1)
            ranking_random.append(rank)

            ranking_original_nbfnet_0.append(int(nbfnet_rank_dict[0][batch_no]))
            ranking_original_nbfnet_1.append(int(nbfnet_rank_dict[1][batch_no]))
            ranking_original_nbfnet_2.append(int(nbfnet_rank_dict[2][batch_no]))
            ranking_original_nbfnet_3.append(int(nbfnet_rank_dict[3][batch_no]))
            ranking_original_nbfnet_4.append(int(nbfnet_rank_dict[4][batch_no]))

    #ranking_dict['Ranking_Original'] = ranking_original
    ranking_dict['AnyBURL'] = ranking_random
    ranking_dict['AnyBURL_NBFNet1'] = ranking_original_nbfnet_0
    ranking_dict['AnyBURL_NBFNet2'] = ranking_original_nbfnet_1
    ranking_dict['AnyBURL_NBFNet3'] = ranking_original_nbfnet_2
    ranking_dict['AnyBURL_NBFNet4'] = ranking_original_nbfnet_3
    ranking_dict['AnyBURL_NBFNet5'] = ranking_original_nbfnet_4
    print('===============================================================================')
    print('AnyBURL_random', ranking_random)
    print('===============================================================================')
    print('AnyBURL_NBFNet', ranking_original_nbfnet_0)
    print('===============================================================================')
    get_metric_score(ranking_dict, metric, num_negatives)
