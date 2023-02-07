#Extract negative triplets for the positive triplets having rules

import os
import random

def gen_ran_neg_ent(entities, all_triplets, triplets_neg, count_tail_access, num_neg):
    num_neg_samples = num_neg
    count_tail_access+=1
    index = random.randint(0, len(entities)-1)
    tail = entities[index]
    triplet_candidate = all_triplets[0]
    triplet = [triplet_candidate[0], triplet_candidate[1], tail]
    if triplet not in all_triplets and triplet not in triplets_neg:
        all_triplets.append(triplet)
        triplets_neg.append(triplet)
    if len(triplets_neg)<num_neg_samples and count_tail_access<len(entities):
        gen_ran_neg_ent(entities, all_triplets, triplets_neg, count_tail_access, num_neg_samples)
    return triplets_neg

def extract_tail_triplets(triplet_candidate, tails, num_neg):
    tail_triplets = []
    tail_triplets_neg = []
    tail_triplets.append(triplet_candidate) # Append the positive triplet
    count_tail_access = 0
    tail_triplets_neg = gen_ran_neg_ent(tails, tail_triplets, tail_triplets_neg, count_tail_access, num_neg)
    return tail_triplets_neg

def extract_head_triplets(triplet_candidate, heads, num_neg):
    triplet_candidate = [triplet_candidate[2], 'INV_' + triplet_candidate[1], triplet_candidate[0]]
    head_triplets = []
    head_triplets_neg = []
    head_triplets.append(triplet_candidate) # Append the positive triplet
    count_tail_access = 0
    head_triplets_neg = gen_ran_neg_ent(heads, head_triplets, head_triplets_neg, count_tail_access, num_neg)
    return head_triplets_neg

def extract_neg(pred_dir, num_neg, pred_type):
    pred = open(os.path.join(pred_dir, pred_type), 'r')
    pred = pred.read().split('\n')[:-1]
    pred_neg = []

    for i in range(len(pred) // 3):
        triplet = pred[i * 3]
        triplet = triplet.split()
        heads = pred[i * 3 + 1].split()
        tails = pred[i * 3 + 2].split()
        heads = heads[1::2]
        tails = tails[1::2]
        # Extract tail triplet batch
        if len(tails)>0:
            triplets_tail = extract_tail_triplets(triplet, tails, num_neg)
            if len(triplets_tail)>0:
                pred_neg.append(triplets_tail)

        if len(heads)>0:
            triplets_head = extract_head_triplets(triplet, heads, num_neg)
            if len(triplets_head)>0:
                pred_neg.append(triplets_head)

    final_neg = []
    for item in pred_neg:
        for i in item:
            i = '\t'.join([str(element) for element in i])
            if i not in final_neg:
                final_neg.append(i)

    return final_neg

