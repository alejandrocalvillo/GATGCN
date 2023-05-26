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
from torch_geometric.nn import ChebConv



#My model (really simple)
class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #2 inputs, 64 as hidden state
        self.conv1 = ChebConv(2, 64, 9)
        self.conv2 = ChebConv(64, 1, 9)

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
        # F.log_softmax(x, dim=1) Cross Entropy(Ver que relaccion tiene una señal con la otra)
