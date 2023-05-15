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
    # Metrics for one setup of CompGCN model, i.e., num_ins, num_context
    train_acc = []
    best_valid_acc = []
    mrr_compgcn = []
    hits1_compgcn = []
    hits3_compgcn = []
    hits10_compgcn = []
    hits10_50_compgcn = []

    # Metrics for one setup of a CompGCN_NBFNet, i.e., num_ins, num_context
    mrr_compgcn_nbfnet = []
    hits1_compgcn_nbfnet = []
    hits3_compgcn_nbfnet = []
    hits10_compgcn_nbfnet = []
    hits10_50_compgcn_nbfnet = []


    for i in range(len(res)):
        if '>>>>>>>>>>>>' in res[i] and 'Early Stopping' not in res[i - 1]:
            t_acc = res[i-1].split(',')[1].split(':')[1]
            val_acc = res[i-1].split(',')[3].split(':')[1]
            train_acc.append(float(t_acc))
            best_valid_acc.append(float(val_acc))
        elif 'MRR for CompGCN :' in res[i]:
            mrr_compgcn.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'Hits@1 for  CompGCN :' in res[i]:
            hits1_compgcn.append(float(res[i].split(':')[1]))
        elif 'Hits@3 for  CompGCN :' in res[i]:
            hits3_compgcn.append(float(res[i].split(':')[1]))
        elif 'Hits@10 for  CompGCN :' in res[i]:
            hits10_compgcn.append(float(res[i].split(':')[1]))
        elif 'Hits@10_50 for CompGCN :' in res[i]:
            hits10_50_compgcn.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'MRR for CompGCN_NBFNet' in res[i]:
            mrr_compgcn_nbfnet.append(float(res[i].split(':')[1].split('(')[1][:-1]))
        elif 'Hits@1 for  CompGCN_NBFNet' in res[i]:
            hits1_compgcn_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@3 for  CompGCN_NBFNet' in res[i]:
            hits3_compgcn_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@10 for  CompGCN_NBFNet' in res[i]:
            hits10_compgcn_nbfnet.append(float(res[i].split(':')[1]))
        elif 'Hits@10_50 for CompGCN_NBFNet' in res[i]:
            hits10_50_compgcn_nbfnet.append(float(res[i].split(':')[1].split('(')[1][:-1]))

    '''
    print('mrr_compgcn', mrr_compgcn)
    print('hits1_compgcn', hits1_compgcn)
    print('hits3_compgcn', hits3_compgcn)
    print('hits10_compgcn', hits10_compgcn)
    print('hits10_50_compgcn', hits10_50_compgcn)
    print('mrr_compgcn_nbfnet', mrr_compgcn_nbfnet)
    print('hits1_compgcn_nbfnet', hits1_compgcn_nbfnet)
    print('hits3_compgcn_nbfnet', hits3_compgcn_nbfnet)
    print('hits10_compgcn_nbfnet', hits10_compgcn_nbfnet)
    print('hits10_50_compgcn_nbfnet', hits10_50_compgcn_nbfnet)
    print('==========================================================================================')
    '''
    #print(train_acc)
    #print(best_valid_acc)

    # Average and output the results in the required format  [MRR, Hits@1, Hits@3, Hits@10, Hits@10_50]
    for i in range(len(mrr_compgcn)//5):
        tr_acc = train_acc[i*5:(i*5+5)]
        v_acc = best_valid_acc[i*5:(i*5+5)]
        mrr = mrr_compgcn[i*5:(i*5+5)]
        hits1 = hits1_compgcn[i*5:(i*5+5)]
        hits3 = hits3_compgcn[i * 5:(i * 5 + 5)]
        hits10 = hits10_compgcn[i * 5:(i * 5 + 5)]
        hits10_50 = hits10_50_compgcn[i * 5:(i * 5 + 5)]

        mrr_nbf = mrr_compgcn_nbfnet[i*25:(i*25+25)]
        hits1_nbf = hits1_compgcn_nbfnet[i*25:(i*25+25)]
        hits3_nbf = hits3_compgcn_nbfnet[i * 25:(i * 25 + 25)]
        hits10_nbf = hits10_compgcn_nbfnet[i * 25:(i * 25 + 25)]
        hits10_50_nbf = hits10_50_compgcn_nbfnet[i * 25:(i * 25 + 25)]



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
            mrr_compgcn_avg = sum_ind/len(non_inf_index)
        else:
            mrr_compgcn_avg = sum(mrr) / 5

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

        print('Model_%d_compgcn' % (i + 1), tr_acc_avg, v_acc_avg)
        print('Model_%d_compgcn' %(i+1), mrr_compgcn_avg, hits1_avg, hits3_avg, hits10_avg, hits10_50_avg)
        print('Model_%d_compgcn_nbfnet' % (i + 1), mrr_nbf_avg, hits1_nbf_avg, hits3_nbf_avg, hits10_nbf_avg,
              hits10_50_nbf_avg)
        print('==========================================================================================')

