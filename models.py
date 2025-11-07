import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import cast
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv

# GCN basic operation single layer
class GraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        add_self=False,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        device=None,
        att=False,
    ):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.weight = nn.Parameter(torch.empty(input_dim, output_dim, device=device))
        if add_self:
            self.self_weight = nn.Parameter(
                torch.empty(input_dim, output_dim, device=device)
            )
        if att:
            self.att_weight = nn.Parameter(torch.empty(input_dim, input_dim, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim, device=device))
        else:
            self.bias = None

        # Initialize parameters after device placement
        init.xavier_uniform_(self.weight)
        if add_self:
            init.xavier_uniform_(self.self_weight)
        if att:
            init.xavier_uniform_(self.att_weight)
        if self.bias is not None:
            init.zeros_(self.bias)

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        # deg = torch.sum(adj, -1, keepdim=True)
        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            # import pdb
            # pdb.set_trace()
            att = x_att @ x_att.permute(0, 2, 1)
            # att = self.softmax(att)
            adj = adj * att

        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y, adj



class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            cast(SAGEConv, conv).reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


