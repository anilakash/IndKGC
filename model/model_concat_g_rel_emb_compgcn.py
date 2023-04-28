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
    def __init__(self, hidden_channels, num_relations, num_rel_graph, num_node_features, num_bases,
                 dropout = 0.1, opn = 'sub', bias=True, act='tanh', num_classes = 2):
        super(CompGCN, self).__init__()
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_rel_graph = num_rel_graph
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout = dropout
        self.opn = opn
        self.bias = bias
        self.act = act
        self.num_classess = num_classes

        ### Work in this file ### Try to synchronise these two classes and methods   ######

        self.rel_emb = nn.Embedding(num_relations, hidden_channels, sparse=False)
        torch.manual_seed(123)
        # Need to check and put all the arguments for CompGCNConv##
        self.conv1 = CompGCNConv(num_node_features, hidden_channels, num_rel_graph)
        self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_rel_graph)
        self.conv3 = CompGCNConv(hidden_channels, hidden_channels, num_rel_graph)
        self.lin = Linear(2*hidden_channels, num_classes)


    def forward(self, data, rel_labels, drop_prob):
        #Obtain relation embedding for the batch
        rel_embs = self.rel_emb(rel_labels)
        rel_graph_emb = get_param((data.len_rel_graph*2, self.p.init_dim))

        r = torch.cat([rel_graph_emb, -rel_graph_emb], dim=0)
        x, r = self.conv1(data.x, data.edge_index, data.edge_type, rel_embed=r)  # node_emb, rel_emb
        #x = self.conv1(data.x, data.edge_index, data.edge_type, data.)
        x = x.relu()
        x, r = self.conv2(x, data.edge_index, data.edge_type, rel_embed=r)
        x = x.relu()
        x = self.conv3(x, data.edge_index, data.edge_type, rel_embed=r)

        x = global_mean_pool(x, data.batch)
        # Concatenate the graph embedding and the target relation embeddings
        x = torch.cat([x, rel_embs], dim=1)
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = self.lin(x)
        return x

