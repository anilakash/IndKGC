import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
#from torch_geometric.nn import global_mean_pool


class RGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_relations, num_node_features, num_bases, num_classes = 2):
        super(RGCN, self).__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_channels, sparse=False)
        torch.manual_seed(123)
        self.conv1 = RGCNConv(num_node_features, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv4 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.lin = Linear(2*hidden_channels, num_classes)


    def forward(self, data, rel_labels, drop_prob):
        #Obtain relation embedding for the batch
        rel_embs = self.rel_emb(rel_labels)
        x = self.conv1(data.x, data.edge_index, data.edge_type)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_type)
        x = x.relu()
        x = self.conv3(x, data.edge_index, data.edge_type)
        x = x.relu()
        x = self.conv4(x, data.edge_index, data.edge_type)

        graph_splits = torch.bincount(data.batch).tolist()
        batch_node_embs = torch.split(x, graph_splits)
        h_batch = []
        t_batch = []
        for graph in batch_node_embs:
            h_batch.append(graph[0,])
            t_batch.append(graph[1,])

        #h_batch = torch.stack(h_batch)  # head embedding
        t_batch = torch.stack(t_batch)  # tail embedding
        #x = global_mean_pool(x, data.batch)  # Graph Embedding
        # x = global_max_pool(x, data.batch)  # Graph Embedding

        # Concatenate the graph embedding and relation embeddings
        x = torch.cat([t_batch, rel_embs], dim=1)

        #x = torch.cat([x, rel_embs], dim=1)
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = self.lin(x)
        return x

