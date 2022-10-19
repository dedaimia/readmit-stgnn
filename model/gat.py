import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from model.gaan import WeightedGATConv

""" 
More efficient implementation from DGL: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
"""
class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_heads,
                 activation=None,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.01,
                 residual=False):
        """
        Args:
            num_layers: number of GAT layers
            in_dim: input feature dim
            num_hidden: hidden size
            num_heads: number of heads in hidden layers
            activation: activation function
            feat_drop: dropout proba for input feature
            attn_drop: dropout proba for attention
            negative_slope: negative slope for leaky ReLU
            residual: whether to use residual connection
        """
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.layers.append(GATConv(
            in_dim, num_hidden, num_heads,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        
        if num_layers > 1:
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GATConv(
                    num_hidden * num_heads, num_hidden, num_heads,
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # last layer only has 1 head
            self.layers.append(GATConv(
                    num_hidden * num_heads, num_hidden, 1,
                    feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.layers[l](g, h).flatten(1)
        return h