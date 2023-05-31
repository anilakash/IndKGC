'''
This program extract rule paths for the given file. The graph would be instantiation of the rules which
satisfies the existence of the given triple.
'''
import random
from multiprocessing import Pool
from collections import defaultdict
from copy import deepcopy


def check_simple_path(rule_ins):
    nodes = []
    flag = 1
    for i in range(len(rule_ins)):
        if i % 2 == 0:
            nodes.append(rule_ins[i])
    for n in nodes:
        if nodes.count(n)>1:
            flag = 0
    return flag


def instantiate_rules(rule, ent, h2r2t, rule_ins, rule_instances, atom_cnt=0):
    rule_ins.append(ent)
    rel = rule[atom_cnt]
    r2t = h2r2t[ent]
    if rel in r2t:
        tails = r2t[rel]
        atom_cnt+=1
        rule_ins.append(rel)
        for t in tails:
            rule_ins_temp = deepcopy(rule_ins)
            if atom_cnt<len(rule):
                instantiate_rules(rule, t, h2r2t, rule_ins_temp, rule_instances, atom_cnt)
            else:
                rule_ins.append(t)
                #Check for repeated nodes
                simple_path_checker = check_simple_path(rule_ins)
                if simple_path_checker == 1:
                    rule_instances.append(rule_ins)
                rule_ins = rule_ins[:-1]
        return rule_instances

def extract_rule_graph(rule, ent, h2r2t):
    head_rel = rule[0]
    rule_body = rule[1]
    rule_ins = []
    rule_instances = []
    rule_examples = instantiate_rules(rule_body, ent, h2r2t, rule_ins, rule_instances, atom_cnt=0)
    if rule_examples != None:
        return rule_examples
    else:
        return []

def get_context(head, tail, h2r2t):   # One-hop context
    context = []
    target_nodes = [head, tail]
    for node in target_nodes:
        root = node
        #print('root', root)
        rel = h2r2t[root]
        #print('rel', rel)
        for r, t in rel.items():
            for i in range(len(t)):
                con_int = []
                con_int.append(root)
                con_int.append(r)
                con_int.append(t[i])
                context.append(con_int)
    return context


def extract_links_on_induced_graph(nodes_induced_graph, h2r2t):
    induced_triplets = []
    for node in nodes_induced_graph:
        root = node
        #print('root', root)
        rel = h2r2t[root]
        #print('rel', rel)
        for r, ts in rel.items():
            for t in ts:
                if t in nodes_induced_graph:
                    induced_triplets.append([root, r, t])

    #print('induced_triplets', induced_triplets)
    #print('------------------------------------')

    return induced_triplets


def extract_path(args):
    rule_list_for_triplet = defaultdict(list)
    lines, rules, h2r2t, num_ins, num_con = args
    triplet, label = lines[0], lines[1][0]
    head, tail, rel = triplet[0], triplet[1], triplet[2]
    triplet_key = str(head) + '\t' + str(tail) + '\t' + str(rel) + '\t' + str(label)
    num_paths = 0
    #context = get_context(head, tail, h2r2t)
    #nodes_induced_graph = []    # Nodes in the induced graph for the given head and tail
    for rule in rules:
        if num_paths < num_ins:
            if rule[0] == rel:
                rule_path_examples = extract_rule_graph(rule, head, h2r2t)
                if len(rule_path_examples)>0:
                    for i in range(len(rule_path_examples)*2):
                        example = random.choice(rule_path_examples)
                        #if example[-1] == tail and example not in rule_list_for_triplet[triplet_key]:
                        if example[-1] == tail:
                            rule_list_for_triplet[triplet_key].append(example)
                            num_paths+=1
                            if num_paths == num_ins:
                                break
                            #print('example', example)
                            #for i in range(len(example)):
                            #    if i % 2 == 0 and example[i] not in nodes_induced_graph:
                            #        nodes_induced_graph.append(example[i])

        else:
            break
    '''
    # Extract all the triplets around the nodes in the induced graph between head and tail
    if num_paths > 0:
        #print('nodes_induced_graph', nodes_induced_graph)
        induced_triplets = extract_links_on_induced_graph(nodes_induced_graph, h2r2t)
        for link in induced_triplets:
            if link not in rule_list_for_triplet[triplet_key]:
                rule_list_for_triplet[triplet_key].append(link)
    
    #print('context', context)
    # Extract context for the given value of num_con
    if num_paths > 0 and num_con > 0:
        #print('Length of context is', len(context))
        cnt_head = 0
        cnt_tail = 0
        unique_con = []
        #print('rule_list_for_triplet', rule_list_for_triplet[triplet_key])
        for i in range(1000):
            con = random.choice(context)
            #unique_con.add(con)
            if cnt_head >= num_con or len(unique_con)==len(context):
                break
            #elif con not in rule_list_for_triplet[triplet_key] and head == con[0]:  # When remove_hops=0, the earlier case
            elif con not in rule_list_for_triplet[triplet_key] and head == con[0] and con[2] != tail:
                rule_list_for_triplet[triplet_key].append(con)
                cnt_head += 1
            if con not in unique_con:
                unique_con.append(con)

        unique_con = []   # For tail's context

        for i in range(1000):
            con = random.choice(context)
            #unique_con.add(con)
            if cnt_tail >= num_con or len(unique_con)==len(context):
                break
            #elif con not in rule_list_for_triplet[triplet_key] and tail == con[0]:  # When remove_hops=0, the earlier case
            elif con not in rule_list_for_triplet[triplet_key] and tail == con[0] and con[2] != head:
                rule_list_for_triplet[triplet_key].append(con)
                cnt_tail += 1
            if con not in unique_con:
                unique_con.append(con)
    #print(rule_list_for_triplet)
    '''
    return rule_list_for_triplet

def triplet_rule_paths(train, rules, h2r2t, num_ins, workers, num_con):
    args = []
    rule_paths_dict = defaultdict(list)
    for lines in train:
        args.append([lines, rules, h2r2t, num_ins, num_con])
    n_proc = workers
    pool = Pool(n_proc)
    rule_paths = pool.map(extract_path, args)
    pool.close()
    pool.join()

    for path_dict in rule_paths:
        for key, val in path_dict.items():
            rule_paths_dict[key] = val
    return rule_paths_dict













