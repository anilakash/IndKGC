import collections
import sys
import os
sys.path.append(os.getcwd())
#from model.model_proj_g_rel_emb_compgcn import *
from model.model_concat_g_rel_emb_compgcn import *
from utils.data_loader import load_train, load_valid, load_test
import operator
import random
import math

def train(train_loader, model, device, optimizer, criterion, drop_prob):
    model.train()
    for data in train_loader:
        rel_labels = data[1].to(device)
        data = data[0].to(device)
        out = model(data, rel_labels, drop_prob)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader, model, device, drop_prob):
     model.eval()
     correct = 0
     for data in loader:
         rel_labels = data[1].to(device)
         data = data[0].to(device)
         out = model(data, rel_labels, drop_prob)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def eval_valid_test(val_or_test_loader, model, device, drop_prob):
    model.eval()
    scores = []
    for data in val_or_test_loader:
        rel_labels = data[1].to(device)
        data = data[0].to(device)
        out = model(data, rel_labels, drop_prob)
        pred = out.argmax(dim=1)
        for index, score in enumerate(out):
            scores.append(float(score[1]))
    return scores

def save_best_model(data_dir, best_val_acc, valid_acc, epoch, best_epoch, model, optimizer, epoch_count, args):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    model_run_no = args.model_run
    if valid_acc > best_val_acc:
        if not os.path.exists(os.path.join(data_dir, 'anyburl_gnn_ranks')):
            os.mkdir(os.path.join(data_dir, 'anyburl_gnn_ranks'))

        torch.save(state, os.path.join(data_dir, 'anyburl_gnn_ranks', model_run_no + '_epoch_%d.pth' % epoch))
        epoch_count = 1
        return valid_acc, epoch, epoch_count
    else:
        epoch_count+=1
        return best_val_acc, best_epoch, epoch_count

def shuffle_batch(batch_score):
    shuffled_batch = {}
    keys = list(batch_score.keys())
    random.shuffle(keys)
    for key in keys:
        shuffled_batch[key] = batch_score[key]
    return shuffled_batch

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

def get_ab_ind_nbfnet_rank(test_strict_ind, nbfnet_score, sorted_batch_score, batch_key):
    anyburl_indices = []
    for key in sorted_batch_score:
        anyburl_indices.append(int(key.split('\t')[1]))
    batch_score = {}
    for index in test_strict_ind:
        key = batch_key.split('\t')[0] + '\t' + str(index) + '\t' + batch_key.split('\t')[2]
        if index in anyburl_indices:
            scr = float(nbfnet_score[index]) + 10000 # To keep AnyBURL predictions at top than nbfnet
            batch_score[key] = float(scr)
        else:
            batch_score[key] = float(nbfnet_score[index])
    sorted_batch_score_final = dict(sorted(batch_score.items(), key=operator.itemgetter(1), reverse=True))
    batch_key_pred_pos = list(sorted_batch_score_final).index(batch_key)
    rank = batch_key_pred_pos+1
    return rank


def get_hybrid_rank(test_strict_ind, nbfnet_score, sorted_batch_score, batch_key):
    anyburl_indices = []
    for key in sorted_batch_score:
        anyburl_indices.append(int(key.split('\t')[1]))

    batch_score = {}
    for index in test_strict_ind:
        key = batch_key.split('\t')[0] + '\t' + str(index) + '\t' + batch_key.split('\t')[2]
        if index in anyburl_indices:
            scr = sorted_batch_score[key] + 10000 # To keep AnyBURL predictions at top
            batch_score[key] = float(scr)
        else:
            batch_score[key] = float(nbfnet_score[index])

    sorted_batch_score_final = dict(sorted(batch_score.items(), key=operator.itemgetter(1), reverse=True))
    batch_key_pred_pos = list(sorted_batch_score_final).index(batch_key)
    rank = batch_key_pred_pos+1

    return rank


def eval_test(data_dir, test_graph, best_epoch, num_negatives, metric, nbfnet_rank_dict, model,
              device, test_strict_ind, nbfnet_score_dict, args):
    model_run_no = args.model_run
    #score_run_no = args.score_run
    drop_prob = float(args.drop_out)
    state = torch.load(os.path.join(data_dir, 'anyburl_gnn_ranks', model_run_no +
                                    '_epoch_%d.pth' % best_epoch), map_location=device)
    model.load_state_dict(state["model"])
    ranking_dict = collections.defaultdict(list)
    # Different Rankings to fetch
    ranking_original = [] # Only the AnyBURL indices

    ranking_random = []   # AnyBURL indices + random rank for non-AnyBURL (RGCN)

    ranking_original_nbfnet_0 = []   # RGCN + NBFNet run 1
    ranking_original_nbfnet_1 = []   # RGCN + NBFNet run 2
    ranking_original_nbfnet_2 = []   # RGCN + NBFNet run 3
    ranking_original_nbfnet_3 = []   # RGCN + NBFNet run 4
    ranking_original_nbfnet_4 = []   # RGCN + NBFNet run 5

    ranking_ab_ind_nbfnet_0 = []     # NBFNet + NBFNet run 1
    ranking_ab_ind_nbfnet_1 = []     # NBFNet + NBFNet run 2
    ranking_ab_ind_nbfnet_2 = []     # NBFNet + NBFNet run 3
    ranking_ab_ind_nbfnet_3 = []     # NBFNet + NBFNet run 4
    ranking_ab_ind_nbfnet_4 = []     # NBFNet + NBFNet run 5

    batch_no = -1
    test_scores = []
    for batch_key, batch_triplets in test_graph.items():
        batch_no += 1
        sample_keys = []
        sample_graphs = []
        for sample_key, sample_graph in batch_triplets.items():
            sample_keys.append(sample_key.split('\t')[:-1])
            sample_graphs.append(sample_graph)
        if len(sample_graphs) > 0:
            test_loader = load_test(sample_graphs)
            scores = eval_valid_test(test_loader, model, device, drop_prob)
            batch_score = {}
            for i in range(len(sample_keys)):
                s_key = str(sample_keys[i][0]) + '\t' + str(sample_keys[i][1]) + '\t' + str(sample_keys[i][2])
                batch_score[s_key] = scores[i]
            # Store the batch_scores
            test_scores.append(batch_score)
            # Suffle the batch_score to break the ties randomly
            batch_score_shuffled = shuffle_batch(batch_score) 
            sorted_batch_score = dict(sorted(batch_score_shuffled.items(), key=operator.itemgetter(1), reverse=True))
            if batch_key in sorted_batch_score: # Case 1: The Pos has a rule
                batch_key_pred_pos = list(sorted_batch_score).index(batch_key)
                ranking_original.append(int(batch_key_pred_pos)+1)
                ranking_random.append(int(batch_key_pred_pos)+1)
                ranking_original_nbfnet_0.append(int(batch_key_pred_pos)+1)
                ranking_original_nbfnet_1.append(int(batch_key_pred_pos)+1)
                ranking_original_nbfnet_2.append(int(batch_key_pred_pos)+1)
                ranking_original_nbfnet_3.append(int(batch_key_pred_pos)+1)
                ranking_original_nbfnet_4.append(int(batch_key_pred_pos)+1)

                # Get ranking of nbfnet on anyburl index at top  for nbfnet + nbfnet
                #print('Working for Case')
                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[0][batch_no],
                                              sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_0.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[1][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_1.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[2][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_2.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[3][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_3.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[4][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_4.append(rank_ab_ind_nbfnet)

            else: # Case 3: Some Negs have rule but Pos does not
                ranking_original.append(num_negatives[batch_no])  # Assigning the least score if no rules
                rank = random.randint(len(sample_graphs)+1, num_negatives[batch_no]+1)
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
                
                # Get ranking of nbfnet on anyburl index at top + nbfnet
                #print('Working for Case 3')
                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[0][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_0.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[1][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_1.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[2][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_2.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[3][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_3.append(rank_ab_ind_nbfnet)

                rank_ab_ind_nbfnet = get_ab_ind_nbfnet_rank(test_strict_ind[batch_no], nbfnet_score_dict[4][batch_no],
                                                            sorted_batch_score, batch_key)
                ranking_ab_ind_nbfnet_4.append(rank_ab_ind_nbfnet)


        else: # Case 2: No rules for Pos as well as Negs.
            ranking_original.append(num_negatives[batch_no])   # Assigning the least score if no rules

            rank = random.randint(0, num_negatives[batch_no]+1)
            ranking_random.append(rank)

            ranking_original_nbfnet_0.append(int(nbfnet_rank_dict[0][batch_no]))
            ranking_original_nbfnet_1.append(int(nbfnet_rank_dict[1][batch_no]))
            ranking_original_nbfnet_2.append(int(nbfnet_rank_dict[2][batch_no]))
            ranking_original_nbfnet_3.append(int(nbfnet_rank_dict[3][batch_no]))
            ranking_original_nbfnet_4.append(int(nbfnet_rank_dict[4][batch_no]))

            ranking_ab_ind_nbfnet_0.append(int(nbfnet_rank_dict[0][batch_no]))
            ranking_ab_ind_nbfnet_1.append(int(nbfnet_rank_dict[1][batch_no]))
            ranking_ab_ind_nbfnet_2.append(int(nbfnet_rank_dict[2][batch_no]))
            ranking_ab_ind_nbfnet_3.append(int(nbfnet_rank_dict[3][batch_no]))
            ranking_ab_ind_nbfnet_4.append(int(nbfnet_rank_dict[4][batch_no]))


    ranking_dict['CompGCN']   = ranking_random
    ranking_dict['CompGCN_NBFNet1'] = ranking_original_nbfnet_0
    ranking_dict['CompGCN_NBFNet2'] = ranking_original_nbfnet_1
    ranking_dict['CompGCN_NBFNet3'] = ranking_original_nbfnet_2
    ranking_dict['CompGCN_NBFNet4'] = ranking_original_nbfnet_3
    ranking_dict['CompGCN_NBFNet5'] = ranking_original_nbfnet_4
    '''
    ranking_dict['NBFNet_NBFNet1'] = ranking_ab_ind_nbfnet_0
    ranking_dict['NBFNet_NBFNet2'] = ranking_ab_ind_nbfnet_1
    ranking_dict['NBFNet_NBFNet3'] = ranking_ab_ind_nbfnet_2
    ranking_dict['NBFNet_NBFNet4'] = ranking_ab_ind_nbfnet_3
    ranking_dict['NBFNet_NBFNet5'] = ranking_ab_ind_nbfnet_4
    '''
    get_metric_score(ranking_dict, metric, num_negatives)

def train_test(data_dir, train_graph, valid_graph, test_graph, num_negatives, nbfnet_rank_dict,
               num_rel, test_strict_ind, nbfnet_score_dict, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    metric = ['mrr', 'hits@1', 'hits@3', 'hits@10', 'hits@10_50']
    num_node_features = int(args.len_rule)*2+4
    num_hidden_channels = int(args.num_hidden)
    lr = float(args.learning_rate)
    num_bases = int(args.num_bases)
    epoch = int(args.epoch)
    patience = int(args.patience)
    drop_prob = float(args.drop_out)
    bias = args.bias
    opn = args.opn
    act = args.act
    concat1 = args.concat1
    concat2 = args.concat2
    concat3 = args.concat3
    concat4 = args.concat4
    projection1 = args.projection1
    projection2 = args.projection2
    tail_only   = args.tail_only


    model = CompGCN(hidden_channels=num_hidden_channels, num_relations=num_rel, num_node_features=num_node_features,
                 num_bases=num_bases, dropout=drop_prob, act=act, opn=opn, bias=bias, concat1=concat1, concat2=concat2,
                    concat3=concat3, concat4=concat4, projection1=projection1, projection2=projection2, tail_only=tail_only)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = load_train(train_graph)
    valid_loader = load_valid(valid_graph)

    best_val_acc = 0.0
    best_epoch = 0
    epoch_count = 0

    for epoch in range(0, epoch):
        train(train_loader, model, device, optimizer, criterion, drop_prob)
        train_acc = test(train_loader, model, device, drop_prob)
        valid_acc = test(valid_loader, model, device, drop_prob)

        best_val_acc, best_epoch, epoch_count = save_best_model(data_dir, best_val_acc, valid_acc, epoch, best_epoch,
                                                                model, optimizer, epoch_count, args) #saving best model
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, '
              f'Best Valid Acc: {best_val_acc:.4f}, Best Epoch: {best_epoch:03d}')

        if epoch_count > patience:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('Using Early Stopping as improvement is same for continuously 10 epochs, No more training needed.')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            break

    eval_test(data_dir, test_graph, best_epoch, num_negatives, metric, nbfnet_rank_dict, model,
              device, test_strict_ind, nbfnet_score_dict, args)
