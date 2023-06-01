# This program will find Average rule instantiation for the given dataset
import os
import pickle as pkl
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='Dataset', default='fb15k237_v1')

args = parser.parse_args()

data_dir = 'data'
#datasets = ['fb15k237_v1', 'fb15k237_v2', 'fb15k237_v3', 'fb15k237_v4',
 #           'WN18RR_v1',  'WN18RR_v2', 'WN18RR_v3',  'WN18RR_v4',
  #          'nell_v1',  'nell_v2', 'nell_v3',  'nell_v4']
datasets = [args.data]
test_name = 'top_5_paths_only/test.pkl'

def decode(batch_key, inst, id2entity, id2rel):
    batch_key = batch_key.split('\t')
    batch_key = [batch_key[0], batch_key[2], batch_key[1]]
    #print(batch_key)
    batch_key[0] = id2entity[int(batch_key[0])]
    batch_key[1] = id2rel[int(batch_key[1])]
    batch_key[2] = id2entity[int(batch_key[2])]
    #print(batch_key)
    batch_key = '  '.join(elements for elements in batch_key)
    #print(batch_key)

    #print(inst)
    inst_decoded = []
    for item in inst:
        temp = []
        for i in range(len(item)):
            if i % 2 == 0:
                temp.append(id2entity[item[i]])
            else:
                temp.append(id2rel[item[i]])
        temp = '  '.join(elements for elements in temp)
        inst_decoded.append(temp)
    #print(inst_decoded)

    print(batch_key, '<--', inst_decoded)


    print('-----------------------------------------------------')

for i in range(len(datasets)):
    test_file = os.path.join(data_dir, datasets[i], test_name)
    entity2id = os.path.join(data_dir, datasets[i], 'entity2id_train_ind.json')
    rel2id = os.path.join(data_dir, datasets[i], 'rel2id.json')
    test = pkl.load(open(test_file, 'rb'))
    entity2id = json.load(open(entity2id))
    rel2id = json.load(open(rel2id))

    id2entity = {val:key for key, val in entity2id.items()}
    id2rel = {val: key for key, val in rel2id.items()}


    num_ins = 0
    for batch_key, sample_graphs in test.items():
        batch_key = batch_key + '\t' + str(1)
        if len(sample_graphs[batch_key]) > 2 and len(sample_graphs[batch_key]) < 6 and \
                int(batch_key.split('\t')[2]) % 2 == 0:
            #print(batch_key, '\t', sample_graphs[batch_key])
            decode(batch_key, sample_graphs[batch_key], id2entity, id2rel)

        num_ins = num_ins + len(sample_graphs[batch_key])
    #print(num_ins/len(test))

