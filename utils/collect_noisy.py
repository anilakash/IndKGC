# This program collects results from all the metrics across different 5 runs and does average.

import argparse
import numpy as np



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', '--result', help='Result File', default='out_fb_top_5_1000_con_tail_rel_5_times')

    args = parser.parse_args()
    result_file = args.result

    # Read file
    res = open(result_file, 'r').read().split('\n')[:-1]

    res_model_runs_5 = []
    # Metrics for one setup of Noisy model, i.e., num_ins, num_context
    train_acc = []
    best_valid_acc = []
    mrr_Noisy = []
    hits1_Noisy = []
    hits3_Noisy = []
    hits10_Noisy = []
    hits10_50_Noisy = []

    # Metrics for one setup of a Noisy_NBFNet, i.e., num_ins, num_context
    mrr_Noisy_nbfnet = []
    hits1_Noisy_nbfnet = []
    hits3_Noisy_nbfnet = []
    hits10_Noisy_nbfnet = []
    hits10_50_Noisy_nbfnet = []


    for i in range(len(res)):
        if 'Epoch: 049' in res[i]:
            t_acc = res[i].split(',')[1].split(':')[1]
            val_acc = res[i].split(',')[3].split(':')[1]
            train_acc.append(float(t_acc))
            best_valid_acc.append(float(val_acc))
        elif '>>>>>>>>>>>>' in res[i] and 'Early Stopping' not in res[i - 1]:
            t_acc = res[i-1].split(',')[1].split(':')[1]
            val_acc = res[i-1].split(',')[3].split(':')[1]
            train_acc.append(float(t_acc))
            best_valid_acc.append(float(val_acc))

        elif 'MRR for Noisy :' in res[i]:
            mrr_Noisy.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'Hits@1 for  Noisy :' in res[i]:
            hits1_Noisy.append(float(res[i].split(':')[1]))
        elif 'Hits@3 for  Noisy :' in res[i]:
            hits3_Noisy.append(float(res[i].split(':')[1]))
        elif 'Hits@10 for  Noisy :' in res[i]:
            hits10_Noisy.append(float(res[i].split(':')[1]))
        elif 'Hits@10_50 for Noisy :' in res[i]:
            hits10_50_Noisy.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'MRR for Noisy_NBFNet' in res[i]:
            mrr_Noisy_nbfnet.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'Hits@1 for  Noisy_NBFNet' in res[i]:
            hits1_Noisy_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@3 for  Noisy_NBFNet' in res[i]:
            hits3_Noisy_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@10 for  Noisy_NBFNet' in res[i]:
            hits10_Noisy_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@10_50 for Noisy_NBFNet' in res[i]:
            hits10_50_Noisy_nbfnet.append(float(res[i].split(':')[1].split('(')[1][:-1]))

    '''
    print('mrr_Noisy', mrr_Noisy)
    print('hits1_Noisy', hits1_Noisy)
    print('hits3_Noisy', hits3_Noisy)
    print('hits10_Noisy', hits10_Noisy)
    print('hits10_50_Noisy', hits10_50_Noisy)
    print('mrr_Noisy_nbfnet', mrr_Noisy_nbfnet)
    print('hits1_Noisy_nbfnet', hits1_Noisy_nbfnet)
    print('hits3_Noisy_nbfnet', hits3_Noisy_nbfnet)
    print('hits10_Noisy_nbfnet', hits10_Noisy_nbfnet)
    print('hits10_50_Noisy_nbfnet', hits10_50_Noisy_nbfnet)
    print('==========================================================================================')
    '''
    print('mrr_Noisy', len(mrr_Noisy))
    print(len(train_acc))
    print(len(best_valid_acc))

    # Average and output the results in the required format  [MRR, Hits@1, Hits@3, Hits@10, Hits@10_50]
    for i in range(len(mrr_Noisy)//5):
        tr_acc = train_acc[i*5:(i*5+5)]
        v_acc = best_valid_acc[i*5:(i*5+5)]
        mrr = mrr_Noisy[i*5:(i*5+5)]
        hits1 = hits1_Noisy[i*5:(i*5+5)]
        hits3 = hits3_Noisy[i * 5:(i * 5 + 5)]
        hits10 = hits10_Noisy[i * 5:(i * 5 + 5)]
        hits10_50 = hits10_50_Noisy[i * 5:(i * 5 + 5)]

        mrr_nbf = mrr_Noisy_nbfnet[i*25:(i*25+25)]
        hits1_nbf = hits1_Noisy_nbfnet[i*25:(i*25+25)]
        hits3_nbf = hits3_Noisy_nbfnet[i * 25:(i * 25 + 25)]
        hits10_nbf = hits10_Noisy_nbfnet[i * 25:(i * 25 + 25)]
        hits10_50_nbf = hits10_50_Noisy_nbfnet[i * 25:(i * 25 + 25)]



        if np.inf in mrr:
            #print('Handling Inf')
            inf_index = set()
            for j in range(len(mrr)):
                if mrr[j] == np.inf:
                    inf_index.add(j)
            #print(inf_index)
            non_inf_index = set.difference(set(range(5)), inf_index)
            #print(non_inf_index)
            sum_ind = 0
            for ind in non_inf_index:
                sum_ind += mrr[ind]
            mrr_Noisy_avg = sum_ind/len(non_inf_index)
        else:
            mrr_Noisy_avg = sum(mrr) / 5

        tr_acc_avg = sum(tr_acc) / 5
        v_acc_avg = sum(v_acc) / 5
        hits1_avg = sum(hits1) / 5
        hits3_avg = sum(hits3) / 5
        hits10_avg = sum(hits10) / 5
        hits10_50_avg = sum(hits10_50) / 5

        mrr_nbf_avg = sum(mrr_nbf)/25
        hits1_nbf_avg = sum(hits1_nbf) / 25
        hits3_nbf_avg = sum(hits3_nbf) / 25
        hits10_nbf_avg = sum(hits10_nbf) / 25
        hits10_50_nbf_avg = sum(hits10_50_nbf) / 25

        print('Model_%d_Noisy' % (i + 1), tr_acc_avg, v_acc_avg)
        print('Model_%d_Noisy' %(i+1), mrr_Noisy_avg, hits1_avg, hits3_avg, hits10_avg, hits10_50_avg)
        print('Model_%d_Noisy_nbfnet' % (i + 1), mrr_nbf_avg, hits1_nbf_avg, hits3_nbf_avg, hits10_nbf_avg,
              hits10_50_nbf_avg)
        print('==========================================================================================')

