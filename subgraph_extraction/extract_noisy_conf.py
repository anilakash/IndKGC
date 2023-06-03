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
        if nodes.count(n) > 1:
            flag = 0
    return flag


def instantiate_rules(rule, ent, h2r2t, rule_ins, rule_instances, atom_cnt=0):
    rule_ins.append(ent)
    rel = rule[atom_cnt]
    r2t = h2r2t[ent]
    if rel in r2t:
        tails = r2t[rel]
        atom_cnt += 1
        rule_ins.append(rel)
        for t in tails:
            rule_ins_temp = deepcopy(rule_ins)
            if atom_cnt < len(rule):
                instantiate_rules(rule, t, h2r2t, rule_ins_temp, rule_instances, atom_cnt)
            else:
                rule_ins.append(t)
                # Check for repeated nodes
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


def get_context(head, tail, h2r2t):  # One-hop context
    context = []
    target_nodes = [head, tail]
    for node in target_nodes:
        root = node
        # print('root', root)
        rel = h2r2t[root]
        # print('rel', rel)
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
        # print('root', root)
        rel = h2r2t[root]
        # print('rel', rel)
        for r, ts in rel.items():
            for t in ts:
                if t in nodes_induced_graph:
                    induced_triplets.append([root, r, t])

    # print('induced_triplets', induced_triplets)
    # print('------------------------------------')

    return induced_triplets


def extract_path(args):
    rule_conf_for_triplet = {}
    conf_acc = []
    lines, rules, h2r2t, num_ins = args
    triplet, label = lines[0], lines[1][0]
    #print('triplet', triplet)
    head, tail, rel = triplet[0], triplet[1], triplet[2]
    triplet_key = str(head) + '\t' + str(tail) + '\t' + str(rel) + '\t' + str(label)
    num_rules = 0

    for rule in rules:
        #if num_rules < num_ins:
        if rule[0] == rel:
            rule_path_examples = extract_rule_graph(rule, head, h2r2t)
            if len(rule_path_examples) > 0:
                for example in rule_path_examples:
                    # if example[-1] == tail and example not in rule_list_for_triplet[triplet_key]:
                    if example[-1] == tail:
                        #rule_list_for_triplet[triplet_key].append(example)
                        conf_acc.append(rule[2])
                        num_rules += 1
                        break


        #else:
        #    break

    # Estimate noisy-or score
    score = 1
    if len(conf_acc)>0:
        for scr in conf_acc:
            score = score * (1.0 - float(scr))
        noisy_or_scr = 1-score
        rule_conf_for_triplet[triplet_key] = noisy_or_scr

    return rule_conf_for_triplet


def triplet_rule_paths(train, rules, h2r2t, num_ins, workers):
    #print('train', train)
    args = []
    rule_conf_dict = defaultdict(list)
    for lines in train:
        args.append([lines, rules, h2r2t, num_ins])
    n_proc = workers
    pool = Pool(n_proc)
    rule_conf = pool.map(extract_path, args)
    pool.close()
    pool.join()

    #print(rule_conf)
    #print('------------------------')

    for conf_dict in rule_conf:
        for key, val in conf_dict.items():
            rule_conf_dict[key] = val
    return rule_conf_dict













