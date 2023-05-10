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

def extract_path(args):
    rule_list_for_triplet = defaultdict(list)
    lines, rules, h2r2t, num_ins = args
    triplet, label = lines[0], lines[1][0]
    head, tail, rel = triplet[0], triplet[1], triplet[2]
    triplet_key = str(head) + '\t' + str(tail) + '\t' + str(rel) + '\t' + str(label)
    num_paths = 0
    context = get_context(head, tail, h2r2t)
    for rule in rules:
        if num_paths < num_ins:
            if rule[0] == rel:
                rule_path_examples = extract_rule_graph(rule, head, h2r2t)
                if len(rule_path_examples)>0:
                    # Get contexts for head and tail

                    for example in rule_path_examples:
                        if example[-1] == tail:
                            rule_list_for_triplet[triplet_key].append(example)
                            num_paths+=1

        else:
            break

    if num_paths > 0:
        #print('Length of context is', len(context))
        cnt_head = 0
        cnt_tail = 0
        for i in range(len(context)):
            con = random.choice(context)
            if con not in rule_list_for_triplet[triplet_key] and head == con[0] and cnt_head < 4:
                rule_list_for_triplet[triplet_key].append(con)
                cnt_head+=1
            elif con not in rule_list_for_triplet[triplet_key] and tail == con[0] and cnt_tail < 4:
                rule_list_for_triplet[triplet_key].append(con)
                cnt_tail += 1
        #print('cnt_head', cnt_head)
        #print('cnt_tail', cnt_tail)
    return rule_list_for_triplet

def triplet_rule_paths(train, rules, h2r2t, num_ins, workers):
    args = []
    rule_paths_dict = defaultdict(list)
    for lines in train:
        args.append([lines, rules, h2r2t, num_ins])
    n_proc = workers
    pool = Pool(n_proc)
    rule_paths = pool.map(extract_path, args)
    pool.close()
    pool.join()

    for path_dict in rule_paths:
        for key, val in path_dict.items():
            rule_paths_dict[key] = val
    return rule_paths_dict













