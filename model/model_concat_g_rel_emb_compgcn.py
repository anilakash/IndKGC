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
                 dropout, act, opn, bias, num_classes = 2):
        super(CompGCN, self).__init__()

        self.num_bases = num_bases
        self.rel_emb = nn.Embedding(num_relations, hidden_channels, sparse=False)

        if self.num_bases > 0:
            self.rel_graph_emb = get_param((self.num_bases, num_node_features))
        else:
            self.rel_graph_emb = get_param((num_relations, num_node_features))
        torch.manual_seed(123)

        if self.num_bases > 0:
            self.conv1 = CompGCNConvBasis(num_node_features, hidden_channels, num_relations, self.num_bases,
                                          dropout, act, opn, bias)
            self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv3 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.lin = Linear(2 * hidden_channels, num_classes)
        else:
            self.conv1 = CompGCNConv(num_node_features, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv3 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.lin = Linear(2*hidden_channels, num_classes)


    def forward(self, data, rel_labels, drop_prob):
        rel_embs = self.rel_emb(rel_labels)
        r = self.rel_graph_emb
        x, r = self.conv1(data.x, data.edge_index, data.edge_type, rel_embed=r)  # node_emb, rel_emb
        x, r = self.conv2(x, data.edge_index, data.edge_type, rel_embed=r)
        x, r = self.conv3(x, data.edge_index, data.edge_type, rel_embed=r)
        x = global_mean_pool(x, data.batch)
        # Concatenate the graph embedding and the target relation embeddings
        x = torch.cat([x, rel_embs], dim=1)
        x = self.lin(x)

        return x

