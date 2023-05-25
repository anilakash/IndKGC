#import torch
from torch_geometric.data import Data
#import pickle
import networkx as nx
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import time
import numpy as np
import torch
import copy

def extract_nodes(triplets):
    nodes = []
    for triplet in triplets:
        if triplet[0] not in nodes:
            nodes.append(triplet[0])
        if triplet[1] not in nodes:
            nodes.append(triplet[1])
    return nodes

def extract_edges(triplets):
    edges = []
    edge_type = []
    for triplet in triplets:
        edges.append(triplet[:2])
        edge_type.append(triplet[-1])
    return edges, edge_type

def rename_nodes(nodes, target_nodes):
    renamed_nodes = {}
    num = 2
    for node in nodes:
        if node == target_nodes[0]:
            renamed_nodes[node] = 0
        elif node == target_nodes[1]:
            renamed_nodes[node] = 1
        else:
            renamed_nodes[node] = num
            num+=1
    #print('renamed nodes ', renamed_nodes)
    renamed_nodes = dict(sorted(renamed_nodes.items(), key=lambda item: item[1]))
    #print('renamed nodes ', renamed_nodes)
    return renamed_nodes

def rename_double_radius(node_double_radius, node_dict):
    renamed_double_radius = {}
    for node, vec in node_double_radius.items():
        node = node_dict[node]
        renamed_double_radius[node] = vec
    return renamed_double_radius

def one_hot(num, max_rule_length):
    vec = np.zeros(max_rule_length)
    vec[num] = 1
    return vec

def get_node_features(G, target_nodes, nodes, max_rule_length):
    #nx.draw_networkx(G)
    #plt.show()
    node_double_radius = {}
    node_features = []
    node_dict = rename_nodes(nodes, target_nodes)
    if not nx.is_empty(G):
        distance_h_u = nx.single_source_shortest_path_length(G, target_nodes[0])
        #print(distance_h_u)
        distance_t_u = nx.single_source_shortest_path_length(G, target_nodes[1])
        #print(distance_t_u)

        # Generate tuples of type [d(u,h), d(u,t)]
        for node in node_dict:
            if node not in target_nodes:
                distance_node_h = distance_h_u[node]
                distance_node_t = distance_t_u[node]
                node_double_radius[node] = [distance_node_h, distance_node_t]
            else:
                if node == target_nodes[0]:
                    node_double_radius[target_nodes[0]] = [0, 1]
                else:
                    node_double_radius[target_nodes[1]] = [1, 0]

        renamed_node_double_radius = rename_double_radius(node_double_radius, node_dict)

        for key, val in renamed_node_double_radius.items():
            val = np.concatenate((one_hot(val[0], max_rule_length), one_hot(val[1], max_rule_length)), axis=0)
            node_features.append(val)
        return node_dict, node_features
    else:
        return node_dict, node_features

def extract_edge_index(edges):
    first_node_list = []
    second_node_list = []
    for edge in edges:
        first_node_list.append(edge[0])
        second_node_list.append(edge[1])
    edge_index = torch.tensor([first_node_list, second_node_list], dtype=torch.long)
    return edge_index


def get_node_features_as_tensors(node_features):
    node_features = np.array(node_features)
    node_features = torch.tensor(node_features, dtype=torch.float)
    return node_features


def add_inverse(edges, edge_type):
    edge_in = copy.deepcopy(edges)
    edge_type_in = copy.deepcopy(edge_type)

    for edge in edges:
        e_in = [edge[1], edge[0]]
        edge_in.append(e_in)

    for etype in edge_type:
        #etype_in = int(etype) + len(set(edge_type))
        if int(etype) % 2 == 0:
            etype_in = int(etype) + 1
        else:
            etype_in = int(etype) - 1
        edge_type_in.append(etype_in)

    return edge_in, edge_type_in

def get_tail_con(tail, num_rel, h2r2t):
    r2t = h2r2t[tail]    # Get all the relational context for tail
    rel2wt = {}
    for rel, ent in r2t.items():
        rel2wt[rel] = len(ent)

    relations = [key for key in rel2wt]
    weight = [val for val in rel2wt.values()]
    z = torch.zeros(num_rel, dtype=torch.float)
    relations = torch.tensor(relations)
    weight = torch.tensor(weight, dtype=torch.float)
    z[relations] = weight
    #z = F.normalize(z, p=2, dim=0)

    return z

def create_valid_graph(rule_graph_for_valid, max_rule_length, num_rel, h2r2t):
    rule_graph = rule_graph_for_valid
    # Generate triplets for the given rule graph
    cnt = 0
    valid_graph = []
    tar_rel = []
    for key, value in rule_graph.items():
        #if cnt < 21:
        #    cnt +=1
        source_triplet = key.split('\t')[:-1]
        source_triplet = [int(element) for element in source_triplet]
        label = key.split('\t')[-1]
        # The rule graphs might not be unique. So take the unique ones [Check...]
        rules_final = []
        for val in value:
            if val not in rules_final:
                rules_final.append(val)
        # Generate triplets from the rule_final
        triplets = []

        for rule in rules_final:
            #print(rule)
            rule_triplets = []
            for i in range(len(rule) // 2):
                h = rule[i * 2]
                r = rule[i * 2 + 1]
                t = rule[i * 2 + 2]
                triplet = [h, t, r]
                if triplet not in rule_triplets:
                    rule_triplets.append(triplet)
            if source_triplet not in rule_triplets:
                for triplet in rule_triplets:
                    if triplet not in triplets:
                        triplets.append(triplet)
        #print('Source Triplet', source_triplet)
        #print('Triplets ', triplets)
        #print('Label', label)
        # Extract nodes, edges, and target nodes for this validing graph

        if len(triplets)>0:
            nodes = extract_nodes(triplets)
            edges, edge_type = extract_edges(triplets)
            target_nodes = [source_triplet[0], source_triplet[1]]
            target_rel = source_triplet[2]
            # Create a Networkx undirected graph
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            #nx.draw_networkx(G)
            #plt.show()

            # Double-Radius Vertex Labeling and Node Features
            node_dict, node_features = get_node_features(G, target_nodes, nodes, max_rule_length)
            #print('Target Nodes ', target_nodes)
            #print('Target Relation', target_rel)
            #print('---------------------------')
            #target_nodes_renamed = [node_dict[target_nodes[0]], node_dict[target_nodes[1]]]
            #print('Target Nodes Renamed ', target_nodes_renamed)
            #print('Final triplets', triplets)
            # Convert the triplets into new node encodings
            for triplet in triplets:
                triplet[0] = node_dict[triplet[0]]
                triplet[1] = node_dict[triplet[1]]
            #print('Node Dict', node_dict)
            #print('Renamed triplets', triplets)

            #print('node features', node_features)

            edges, edge_type = extract_edges(triplets)
            # Add inverse edges and inverse edge_types
            edges, edge_type = add_inverse(edges, edge_type)
            #Extract PyG data attributes
            x = get_node_features_as_tensors(node_features)
            edge_index = extract_edge_index(edges)
            edge_type = torch.tensor(edge_type)
            y = torch.tensor([int(label)], dtype=torch.long)
            z = get_tail_con(target_nodes[1], num_rel, h2r2t)  # Get the vector giving relational context to tail
            data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
            valid_graph.append([data, target_rel, z])

    return valid_graph

