#System
import os

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

#Pythorch Geometric
import torch_geometric
from torch_geometric.nn import ChebConv, GCNConv



class MyGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(MyGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.conv_layers.append(GCNConv(hidden_dim, output_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norms[i](x)
            x = self.dropout(x)

        x = self.conv_layers[-1](x, edge_index)

        return x

        # F.log_softmax(x, dim=1) Cross Entropy(Ver que relaccion tiene una señal con la otra)
