import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_mean_pool
from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis


class CompGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_relations,  num_node_features, num_bases,
                 dropout, opn, bias, num_classes = 2):
        super(CompGCN, self).__init__()
        self.act = torch.tanh
        #self.num_node_features = num_node_features
        #self.hidden_channels = hidden_channels
        #self.num_rel_graph = num_rel_graph
        #self.num_relations = num_relations
        #self.num_bases = num_bases
        #self.dropout = dropout
        #self.opn = opn
        #self.bias = bias
        #self.act = act
        #self.num_classess = num_classes

        ### Work in this file ### Try to synchronise these two classes and methods   ######

        self.rel_emb = nn.Embedding(num_relations, hidden_channels, sparse=False)
        self.rel_graph_emb = get_param((num_relations, num_node_features))
        torch.manual_seed(123)
        # Need to check and put all the arguments for CompGCNConv##
        self.conv1 = CompGCNConv(num_node_features, hidden_channels, num_relations, dropout, opn, bias, self.act)
        self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, opn, bias, self.act)
        self.conv3 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, opn, bias, self.act)
        self.lin = Linear(2*hidden_channels, num_classes)


    def forward(self, data, rel_labels, drop_prob):
        rel_embs = self.rel_emb(rel_labels)
        r = self.rel_graph_emb
        #print('rel emb in model', r.shape)
        x, r = self.conv1(data.x, data.edge_index, data.edge_type, rel_embed=r)  # node_emb, rel_emb
        #print('I am working till first conv')
        #x = self.conv1(data.x, data.edge_index, data.edge_type, data.)
        x = x.relu()
        x, r = self.conv2(x, data.edge_index, data.edge_type, rel_embed=r)
        #print('I am working till 2nd conv')
        #print('2', x.shape, r.shape)
        x = x.relu()
        #print('3', x.shape, r.shape)
        x, r = self.conv3(x, data.edge_index, data.edge_type, rel_embed=r)
        #print('4', x.shape, r.shape)
        #print('===============================================================')
        x = global_mean_pool(x, data.batch)
        # Concatenate the graph embedding and the target relation embeddings
        x = torch.cat([x, rel_embs], dim=1)
        #x = F.dropout(x, p=drop_prob, training=self.training)
        x = self.lin(x)

        return x

