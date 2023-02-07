
import os


def extract_tail_triplets(triplet_candidate, tails, entity2id, rel2id):
    tail_triplets = {}
    triplet_can_key = str(entity2id[triplet_candidate[0]]) + '\t' + str(entity2id[triplet_candidate[2]]) + \
                      '\t' + str(rel2id[triplet_candidate[1]])

    if len(tails)>0:  # If Tails is empty, it will return empty (Case 1)
        for i in range(len(tails)//2):
            triplet_key = str(entity2id[triplet_candidate[0]]) + '\t' + str(entity2id[tails[i*2]]) \
                          + '\t' + str(rel2id[triplet_candidate[1]])
            triplet_score = tails[i*2+1]
            #triplet_dict = {}
            #triplet_dict[triplet_key] = triplet_score
            tail_triplets[triplet_key]=triplet_score # Adding all the triplets with possible tails and socre
        return triplet_can_key, tail_triplets
        #print('===================================')
    else:
        return triplet_can_key, tail_triplets

def extract_head_triplets(triplet_candidate, heads, entity2id, rel2id):
    head_triplets = {}
    triplet_rev_key = str(entity2id[triplet_candidate[2]]) + '\t' + str(entity2id[triplet_candidate[0]]) + \
                      '\t' + str(rel2id['INV_' + triplet_candidate[1]])
    if len(heads)>0: # If Tails is empty, it will return empty (Case 1)
        #print(heads)
        for i in range(len(heads) // 2):
            triplet_key = str(entity2id[triplet_candidate[2]]) + '\t' + str(entity2id[heads[i*2]]) + '\t' +\
                          str(rel2id['INV_' + triplet_candidate[1]])
            #if triplet not in head_triplets:
            triplet_score = heads[i * 2 + 1]
            #triplet_dict = {}
            #triplet_dict[triplet_key] = triplet_score
            head_triplets[triplet_key]=triplet_score # Adding all the triplets with possible tails
        return triplet_rev_key, head_triplets
    else:
        return triplet_rev_key, head_triplets

def generate_batch_triplets_with_scores(pred_test_dir, pred_type, entity2id, rel2id):
    pred_test = open(os.path.join(pred_test_dir, pred_type), 'r')
    test = pred_test.read().split('\n')[:-1]
    test_ind_triplet_batches = {}
    for i in range(len(test) // 3):
        triplet = test[i * 3]
        triplet = triplet.split()
        heads = test[i * 3 + 1].split()
        tails = test[i * 3 + 2].split()
        heads = heads[1:]
        tails = tails[1:]
        # Extract tail triplet batch
        triplet_can_key, triplets_tail = extract_tail_triplets(triplet, tails, entity2id, rel2id)
        triplet_rev_key, triplets_head = extract_head_triplets(triplet, heads, entity2id, rel2id)
        test_ind_triplet_batches[triplet_can_key] = triplets_tail
        test_ind_triplet_batches[triplet_rev_key] = triplets_head
    return test_ind_triplet_batches


