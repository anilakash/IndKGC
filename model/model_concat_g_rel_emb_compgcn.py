import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis


class CompGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_relations,  num_node_features, num_bases,
                 dropout, act, opn, bias, concat0, concat1, concat2, concat3, concat4,
                 projection1, projection2, tail_only, num_classes = 2):
        super(CompGCN, self).__init__()

        self.num_bases = num_bases
        self.concat0 = concat0
        self.concat1 = concat1
        self.concat2 = concat2
        self.concat3 = concat3
        self.concat4 = concat4
        self.projection1 = projection1
        self.projection2 = projection2
        self.tail_only = tail_only

        #self.rel_emb = nn.Embedding(num_relations, hidden_channels, sparse=False)

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
            self.conv4 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            #self.conv5 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            #self.conv6 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.lin = Linear(2 * hidden_channels, num_classes)
        else:
            self.conv1 = CompGCNConv(num_node_features, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv3 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            self.conv4 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            #self.conv5 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            #self.conv6 = CompGCNConv(hidden_channels, hidden_channels, num_relations, dropout, act, opn, bias)
            if self.concat2:
                self.lin = Linear(4*hidden_channels + num_relations, num_classes) # con(G, R, H, T)
            elif self.projection1 or self.projection2 or self.tail_only:  # G*R or G * |R - |H-T|| or tail_only
                self.lin = Linear(hidden_channels + num_relations, num_classes)
            else:  # Other concatenations
                #self.lin = Linear(2 * hidden_channels + num_relations, num_classes) # Use with tail rel contexts
                self.lin = Linear(2 * hidden_channels,  num_classes)


    def forward(self, data, rel_labels, z, drop_prob):
        #print('z', z)
        r = self.rel_graph_emb
        x, r = self.conv1(data.x, data.edge_index, data.edge_type, rel_embed=r)  # node_emb, rel_emb
        x, r = self.conv2(x, data.edge_index, data.edge_type, rel_embed=r)
        x, r = self.conv3(x, data.edge_index, data.edge_type, rel_embed=r)
        x, r = self.conv4(x, data.edge_index, data.edge_type, rel_embed=r)
        #x, r = self.conv5(x, data.edge_index, data.edge_type, rel_embed=r)
        #x, r = self.conv6(x, data.edge_index, data.edge_type, rel_embed=r)

        graph_splits = torch.bincount(data.batch).tolist()
        batch_node_embs = torch.split(x, graph_splits)
        h_batch = []
        t_batch = []
        for graph in batch_node_embs:
            h_batch.append(graph[0,])
            t_batch.append(graph[1,])

        h_batch = torch.stack(h_batch)    # head embedding
        t_batch = torch.stack(t_batch)    # tail embedding
        x = global_mean_pool(x, data.batch)  # Graph Embedding
        #x = global_max_pool(x, data.batch)  # Graph Embedding
        rel_embs = torch.index_select(r, 0, rel_labels)   # Relation Embedding
        #diff_h_t = (h_batch - t_batch) * (h_batch - t_batch)

        if self.concat0:
            #x = torch.cat([t_batch, rel_embs, z], dim=1)
            x = torch.cat([t_batch, rel_embs], dim=1)
            x = self.lin(x)
        elif self.concat1:
            x = torch.cat([x, rel_embs, z], dim=1)
            x = self.lin(x)
        elif self.concat2:
            x = torch.cat([x, rel_embs, h_batch, t_batch, z], dim=1)
            x = self.lin(x)
        elif self.concat3:
            x = torch.cat([x, (t_batch - h_batch) * rel_embs, z], dim=1)
            x = self.lin(x)
        elif self.concat4:
            x = torch.cat([x, (t_batch * h_batch) * rel_embs, z], dim=1)
            x = self.lin(x)
        elif self.projection1:
            x = torch.cat([x * rel_embs, z], dim=1)
            x = self.lin(x)
        elif self.projection2:
            x = torch.cat([x * (rel_embs - h_batch - t_batch), z], dim=1)
            x = self.lin(x)
        elif self.tail_only:
            x = torch.cat([t_batch, z], dim=1)
            x = self.lin(x)
        else:
            print('Not Implemented...')
        return x

