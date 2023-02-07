#Program to encode train, valid, train_ind, test_ind, and rules
import json
import os
import random
from collections import defaultdict




def encoding(triplet_type, entity2id, rel2id):
    triplets = []
    for item in triplet_type:
        item_mapped = [entity2id[item[0]], entity2id[item[2]], rel2id[item[1]]]
        triplets.append(item_mapped)
    return triplets

def add_inverse_rel(data):
    data_with_inv = []
    for triple in data:
        data_with_inv.append(triple)
        inv_triple = [triple[2], 'INV_'+triple[1], triple[0]]
        data_with_inv.append(inv_triple)
    return data_with_inv

#The below function is redundant. Merge this function with the sucessor encode.
def encode_valid(data_dir, valid_file, entity2id, rel2id):
    valid = [line.split() for line in valid_file.read().split('\n')[:-1]]
    valid = add_inverse_rel(valid)
    encoded_valid = encoding(valid, entity2id, rel2id)
    return encoded_valid

def encode_train_ind(data_dir, train_ind_file):
    entity2id = {}
    train_ind = [line.split() for line in train_ind_file.read().split('\n')[:-1]]
    train_ind = add_inverse_rel(train_ind)
    #Get Relation encoding from Training.txt [Relations in the Inductive dataset are subset of Main Training dataset]
    rel2id = json.load(open(data_dir + '/rel2id.json', 'r'))
    # Encode entities
    ent = 0
    for item in train_ind:
        if item[0] not in entity2id:
            entity2id[item[0]] = ent
            ent += 1
        if item[2] not in entity2id:
            entity2id[item[2]] = ent
            ent += 1
    with open(os.path.join(data_dir, 'entity2id_train_ind.json'), 'w') as f:
        json.dump(entity2id, f)
    encoded_train_ind = encoding(train_ind, entity2id, rel2id)
    return encoded_train_ind, entity2id, rel2id

def encode_valid_pos_neg(val_pos_file, val_neg, entity2id, rel2id):
    val_pos = [line.split() for line in val_pos_file.read().split('\n')[:-1]]
    val_pos = add_inverse_rel(val_pos)
    val_neg = [line.split('\t') for line in val_neg]
    encoded_val_pos = encoding(val_pos, entity2id, rel2id)
    encoded_val_neg = encoding(val_neg, entity2id, rel2id)
    return encoded_val_pos, encoded_val_neg

def encode(data_dir, train_pos_file, train_neg):
    entity2id = {}
    rel2id = {}
    train_pos = [line.split() for line in train_pos_file.read().split('\n')[:-1]]
    #train_neg = [line.split() for line in train_neg_file.read().split('\n')[:-1]]
    train_pos = add_inverse_rel(train_pos)
    train_neg = [line.split('\t') for line in train_neg]
    #Encode relations and entities
    ent = 0
    rel = 0
    for item in train_pos:
        if item[0] not in entity2id:
            entity2id[item[0]] = ent
            ent += 1
        if item[2] not in entity2id:
            entity2id[item[2]] = ent
            ent += 1
        if item[1] not in rel2id:
            rel2id[item[1]] = rel
            rel += 1

    with open(os.path.join(data_dir, 'rel2id.json'), 'w') as f:
        json.dump(rel2id, f)

    with open(os.path.join(data_dir, 'entity2id_train.json'), 'w') as f:
        json.dump(entity2id, f)

    encoded_train_pos = encoding(train_pos, entity2id, rel2id)
    encoded_train_neg = encoding(train_neg, entity2id, rel2id)

    return encoded_train_pos, encoded_train_neg, entity2id, rel2id

#Generate negative samples for training data
def generate_neg_samples(train, no_of_neg_sample_per_edge):
    entity_list = []
    neg_edges = []
    pos_edges = train
    for triplet in train:
        if triplet[0] not in entity_list:
            entity_list.append(triplet[0])
        if triplet[1] not in entity_list:
            entity_list.append(triplet[1])

    for edge in pos_edges:
        rel = edge[2]
        head = edge[0]
        tail = edge[1]
        #generate possible head and tail
        head_pos = []
        tail_pos = []
        for h in range(len(entity_list)):
            if h != head and h!= tail:
                head_pos.append(h)
                tail_pos.append(h)

        for i in range(len(tail_pos)):
            tail_temp = random.choice(tail_pos)
            if [head, tail_temp, rel] not in pos_edges and head != tail_temp:
                if [head, tail_temp, rel] not in neg_edges:
                    neg_edges.append([head, tail_temp, rel])
                    break

    # Put labels in positive and negative edges
    train_pos = []
    train_neg = []
    for edge in pos_edges:
        edge = [edge, [1]]
        train_pos.append(edge)

    for edge in neg_edges:
        edge = [edge, [0]]
        train_neg.append(edge)
    return train_pos, train_neg

#Generate entity to relation to entity dictionary (or {head: {relation: [tail1, tail2, ...]}})
def entity2rel2entity(train):
    h2rt = defaultdict(list)
    h2r2t = defaultdict(dict)
    for triplet in train:
        h = triplet[0]
        t = triplet[1]
        r = triplet[2]
        h2rt[h].append((r,t))

    for h, rt in h2rt.items():
        r2t = defaultdict(list)
        for items in rt:
            r2t[items[0]].append(items[1])
        h2r2t[h].update(r2t)
    return h2r2t


def put_labels(train_pos, train_neg):
    train_pos_label = []
    train_neg_label = []
    for triplet in train_pos:
        train_pos_label.append([triplet, [1]])

    for triplet in train_neg:
        train_neg_label.append([triplet, [0]])

    return train_pos_label, train_neg_label




