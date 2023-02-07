#This program will extract the rules and encode using the rel2id

import json
import os.path
import re
import pandas as pd
from copy import deepcopy

from collections import defaultdict

def restructure(atom):
    temp = atom[0]
    atom[0] = atom[1]
    atom[1] = atom[2]
    atom[2] = temp
    return atom

def inverse_atom(atom):
    temp = atom[0]
    atom[0] = atom[1]
    atom[1] = temp
    atom[2] = 'INV_' + atom[2]
    return atom

def add_or_rem_INV_(item):
    if item[2].startswith('INV_'):
        item[2] = item[2].split('INV_')[1]
    else:
        item[2] = 'INV_' + item[2]
    return item


def get_rule_reversed(body):
    body_temp = deepcopy(body)
    body_rev = []
    body_max = len(body_temp)
    for i in range(1, body_max+1):
        b = body_temp[body_max-i]
        # Reversing only the relation sequences as the variables do not matter now
        # Variables matter while encoding the original rule not the rule in inverse
        b = add_or_rem_INV_(b)
        body_rev.append(b)
    return body_rev


def enc(rule_file, data_dir):
    rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'rb'))
    rules = pd.read_csv(rule_file, sep='\t', header=None)
    rules = rules.sample(frac=1).sort_values([2], ascending=False) # The rules are sorted
    rules = rules[3].tolist()

    #rules = [line for line in rule_file.read().split('\n')[:-1]]
    #rules = [line.split('\t')[3] for line in rule_file.read().split('\n')[:-1]]
    encoded_rules = []
    for line in rules:
        line = re.sub(r'\(', ' ', line)
        line = re.sub(r'\)', ' ', line)
        line = re.sub(',', ' ', line)
        line = line.split('<=')
        head = line[0].split()
        body = line[1].split('  ')
        body = [b.split() for b in body]
        #Putting in format <head, tail, relation>
        head = restructure(head)
        for b in body:
            b = restructure(b)

        #Put the inverse relation in the rule's body
        for i in range(len(body)):
            atom = body[i]
            if len(body)==1 and atom[0] > atom[1]:
                atom = inverse_atom(atom)
            elif i==0 and atom[0] < atom[1] and len(body)>1:
                atom = inverse_atom(atom)
            elif i!= 0 and atom[0] > atom[1]:
                atom = inverse_atom(atom)

        body_rev = get_rule_reversed(body)
        head_rel_rev = 'INV_' + head[2]
        # Get encodings
        head[2] = rel2id[head[2]]
        head_rel_rev = rel2id[head_rel_rev]

        for b in body:
            b[2] = rel2id[b[2]]

        for b in body_rev:
            b[2] = rel2id[b[2]]

        #Generating relations in head and relation sequences in body
        head_rel = head[2]
        body_rel_seq = []
        body_rel_seq_rev = []
        for b in body:
            body_rel_seq.append(b[2])

        for b in body_rev:
            body_rel_seq_rev.append(b[2])

        #Store all the relation sequences corresponding to each rule body which yield relation in head
        encoded_rules.append((head_rel, body_rel_seq)) # A tuple of head's relation and body
        encoded_rules.append((head_rel_rev, body_rel_seq_rev)) # A tuple of reversed head's relation and body
    # There are some rule bodies which may appear more than once, pick the one with highest confidence.
    # As the rules are already sorted in descending order, keeping the first occurrence does the above task.
    rules_final = []
    for index, item in enumerate(encoded_rules):
        if item not in rules_final:
            rules_final.append(item)
    return rules_final





