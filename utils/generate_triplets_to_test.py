# Generate test batches as per AnyBURL test prediction for inductive test
from collections import defaultdict
import pickle as pkl
import os


def extract_tail_triplets(triplet_candidate, tails):
    tail_triplets = []
    if len(tails)>0:  # If Tails is empty, it will return empty (Case 1)
        for t in tails:
            triplet = [triplet_candidate[0], triplet_candidate[1], t]
            if triplet not in tail_triplets:
                tail_triplets.append(triplet) # Adding all the triplets with possible tails
        # Add if source triplet not present in tail_triplets (Case 2)
        if triplet_candidate not in tail_triplets:
            tail_triplets.append(triplet_candidate)
        return tail_triplets
    else:
        return tail_triplets

def extract_head_triplets(triplet_candidate, heads):
    head_triplets = []
    triplet_rev = [triplet_candidate[2], 'INV_' + triplet_candidate[1], triplet_candidate[0]]
    if len(heads)>0: # If Tails is empty, it will return empty (Case 1)
        for h in heads:
            triplet = [triplet_candidate[2], 'INV_' + triplet_candidate[1], h]
            if triplet not in head_triplets:
                head_triplets.append(triplet) # Adding all the triplets with possible tails
        # Add if source triplet when reversed not present in head_triplets (Case 2)
        if triplet_rev not in head_triplets:
            head_triplets.append(triplet_rev)
        return head_triplets, triplet_rev
    else:
        return head_triplets, triplet_rev

def generate_batch_triplets(pred_test_dir, pred_type):
    pred_test = open(os.path.join(pred_test_dir, pred_type), 'r')
    test = pred_test.read().split('\n')[:-1]
    test_ind_triplet_batches = {}
    for i in range(len(test) // 3):
        triplet = test[i * 3]
        triplet = triplet.split()
        #print(triplet)
        heads = test[i * 3 + 1].split()
        tails = test[i * 3 + 2].split()
        heads = heads[1::2]
        tails = tails[1::2]
        # Extract tail triplet batch
        triplets_tail = extract_tail_triplets(triplet, tails)
        triplets_head, triplet_rev = extract_head_triplets(triplet, heads)
        test_ind_triplet_batches[triplet[0] + '\t' + triplet[1] + '\t' + triplet[2]] = triplets_tail
        test_ind_triplet_batches[triplet_rev[0] + '\t' + triplet_rev[1] + '\t' + triplet_rev[2]] = triplets_head
    return test_ind_triplet_batches


