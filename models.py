import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import cast
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Third Message Passing Layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Fourth Message Passing Layer
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = self.out(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, dropout: float = 0.5):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphSAGE requires at least two layers for node classification.")

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(dataset.num_features, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, dataset.num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            cast(SAGEConv, conv).reset_parameters()

    def forward(self, data_or_x, edge_index=None):
        if edge_index is None:
            data = data_or_x
            x, edge_index = data.x, data.edge_index
        else:
            x = data_or_x
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


