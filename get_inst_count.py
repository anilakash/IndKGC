# This program will find Average rule instantiation for the given dataset
import os
import pickle as pkl

data_dir = 'data'
datasets = ['fb15k237_v1', 'fb15k237_v2', 'fb15k237_v3', 'fb15k237_v4',
            'WN18RR_v1',  'WN18RR_v2', 'WN18RR_v3',  'WN18RR_v4',
            'nell_v1',  'nell_v2', 'nell_v3',  'nell_v4']

test_name = 'top_5_paths_context_0_hop_1/test.pkl'

for i in range(len(datasets)):
    test_file = os.path.join(data_dir, datasets[i], test_name)
    test = pkl.load(open(test_file, 'rb'))

    num_ins = 0
    for batch_key, sample_graphs in test.items():
        batch_key = batch_key + '\t' + str(1)
        #print(len(sample_graphs[batch_key]))
        num_ins = num_ins + len(sample_graphs[batch_key])

    print(num_ins/len(test))

