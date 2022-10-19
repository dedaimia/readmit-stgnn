### Refered to https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn.py
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 n_layers,
                 activation,
                 dropout=0.):
        """
        Args:
            g: DGL graph instance
            in_feats: number of input features
            n_hidden: number of hidden units
            n_classes: number of output classes
            n_layers: number of GCN layers (excluding output layer)
            activation: activation function
            dropout: dropout proba
        """
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        """
        Args:
            features: feature at one time step, shape (num_nodes, in_dim)
        Returns:
            h: output feature, shape (num_nodes, hidden_dim)
        """
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h