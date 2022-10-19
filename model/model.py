import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import dgl
from dgl.utils import expand_as_pair

sys.path.append("../")
import utils
import math
import numpy as np
from scipy import linalg as la
from pytorch_tabnet import tab_network
import copy

# from model.gat import GATLayer,
from model.gat import GAT
from model.gcn import GCN
from model.graphsage import GraphSAGE
from model.tcn import TemporalConvNet
from model.gaan import GatedGAT
from model.gin import GIN
from model.transformer import TransformerEncoderLayer
from torch.nn.parameter import Parameter
# from model.hippo_code.op import transition
# from model.hippo_code.components import get_activation, get_initializer
# from model.hippo_code.memory import (
#     forward_aliases,
#     backward_aliases,
#     bilinear_aliases,
#     zoh_aliases,
# )

import tqdm


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor"""
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x) for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor"""
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(xs, dim) if isinstance(xs[0], torch.Tensor) else xs[0])
            for xs in zip(*tups)
        )
    else:
        return torch.cat(tups, dim)


class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: shape (batch, hidden, seq_len)
        """
        # input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        t3 = x[:, :, -1]

        out = torch.cat(
            [t1.squeeze(-1), t2.squeeze(-1), t3], 1
        )  # shape (batch, 3 * hidden_size)
        return out


class GConvLayers(nn.Module):
    """
    Multi-layer GCN/GAT/Multi-head GAT/GraphSAGE/Gated GAT
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        g_conv="graphsage",
        activation_fn="relu",
        norm=None,
        dropout=0.0,
        device=None,
        is_classifier=False,
        ehr_encoder_name=None,
        cat_dims=[],
        cat_idxs=[],
        cat_emb_dim=1,
        **kwargs,
    ):
        super(GConvLayers, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.g_conv = g_conv
        self.activation_fn = activation_fn
        self.device = device
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.is_classifier = is_classifier

        # ehr encoder
        if ehr_encoder_name is not None:
            if ehr_encoder_name == "embedder":
                print("Using embedder to embed ehr data...")
                self.embedder = tab_network.EmbeddingGenerator(
                    input_dim=in_dim,
                    cat_dims=cat_dims,
                    cat_idxs=cat_idxs,
                    cat_emb_dim=cat_emb_dim,
                )
                in_dim = (in_dim - len(cat_idxs)) + len(cat_idxs) * cat_emb_dim
            else:
                raise NotImplementedError
        else:
            self.embedder = None

        self.layers = nn.ModuleList()
        if g_conv == "multihead_gat":
            multihead_gat = GAT(
                num_layers=num_gcn_layers,
                in_dim=in_dim,
                num_hidden=hidden_dim,
                num_heads=kwargs["num_heads"],
                activation=None,  # we wil add activation later in forward
                feat_drop=dropout,
                attn_drop=dropout,
                negative_slope=kwargs["negative_slope"],
                residual=kwargs["gat_residual"],
            )
            self.layers = multihead_gat.gat_layers

        elif g_conv == "gat":
            gat = GAT(
                num_layers=num_gcn_layers,
                in_dim=in_dim,
                num_hidden=hidden_dim,
                num_heads=1,
                activation=None,  # we wil add activation later in forward
                feat_drop=dropout,
                attn_drop=dropout,
                negative_slope=kwargs["negative_slope"],
                residual=kwargs["gat_residual"],
            )
            self.layers = gat.gat_layers

        elif g_conv == "graphsage":
            # we will add activation later in forward
            graphsage = GraphSAGE(
                in_feats=in_dim,
                n_hidden=hidden_dim,
                n_layers=num_gcn_layers,
                activation=None,
                norm=norm,
                dropout=dropout,
                aggregator_type=kwargs["aggregator_type"],
            )
            self.layers = graphsage.layers
        elif g_conv == "gaan":
            self.layers.append(
                GatedGAT(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    map_feats=kwargs["gaan_map_feats"],
                    num_heads=kwargs["num_heads"],
                    activation=None,  # we wil add activation later in forward
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=kwargs["negative_slope"],
                    residual=kwargs["gat_residual"],
                )
            )
            for _ in range(1, num_gcn_layers):
                self.layers.append(
                    GatedGAT(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        map_feats=kwargs["gaan_map_feats"],
                        num_heads=kwargs["num_heads"],
                        activation=None,  # we wil add activation later in forward
                        feat_drop=dropout,
                        attn_drop=dropout,
                        negative_slope=kwargs["negative_slope"],
                        residual=kwargs["gat_residual"],
                    )
                )
        elif g_conv == "gin":
            self.layers = GIN(
                num_layers=num_gcn_layers,
                num_mlp_layers=kwargs["num_mlp_layers"],
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                learn_eps=kwargs["learn_eps"],
                neighbor_pooling_type=kwargs["neighbor_pooling_type"],
            )

        else:
            # we will add activation later in forward
            gcn = GCN(
                in_dim, hidden_dim, num_gcn_layers, activation=None, dropout=dropout
            )
            self.layers = gcn.layers

        # optionally for non-temporal models
        if self.is_classifier:
            self.fc = nn.Linear(hidden_dim, kwargs["num_classes"])

    def forward(self, g, inputs):
        """
        Args:
            inputs: shape (batch, in_dim)
        Returns:
            h: shape (batch, hidden_dim) using "mean" aggregate or (batch, hidden_dim*num_heads) using
                "cat" aggregate
        """

        h = inputs

        if self.embedder is not None:
            h = self.embedder(h)

        if self.g_conv != "gin":
            for i in range(self.num_gcn_layers):
                h = self.layers[i](g, h)

                # NEW
                if self.g_conv == "gat" or self.g_conv == "multihead_gat":
                    h = h.flatten(1)

                if i != self.num_gcn_layers - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
        else:
            h = self.layers(g, h)

        if self.is_classifier:
            logits = self.fc(h)
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return logits, h
        else:
            return h


class GConvGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        norm=None,
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        self.norm = (norm,)
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        # gconv_gate includes reset and update gates, that's why hidden_dim * 2
        self.gconv_gate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            norm=norm,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        self.gconv_candidate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            norm=norm,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim * 2,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

    def forward(self, graph, inputs, state):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # reset and update gates
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv_gate(graph, inputs_state)  # (num_nodes, hidden_dim*2)
        if self.add_bias:
            h = h + self.gate_bias
        h = torch.sigmoid(h)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(h, split_size_or_sections=self.hidden_dim, dim=-1)

        # candidate
        c = self.gconv_candidate(
            graph, torch.cat([inputs, r * state], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        new_state = u * state + (1 - u) * c

        return new_state


class GConvMGUCell(nn.Module):
    """
    Graph Convolution Minimal Gated Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvMGUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.gconv_gate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        self.gconv_candidate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

    def forward(self, graph, inputs, state):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # forget gate
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        f = self.gconv_gate(graph, inputs_state)  # (num_nodes, hidden_dim)
        if self.add_bias:
            f = f + self.gate_bias
        f = torch.sigmoid(f)

        # candidate
        c = self.gconv_candidate(
            graph, torch.cat([inputs, f * state], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        new_state = f * c + (1 - f) * state

        return new_state


class GConvMinimalRNNCell(nn.Module):
    """
    Graph Convolution Minimal Gated Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvMinimalRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.gconv_z = GConvLayers(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        self.gconv_u = GConvLayers(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.z_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.z_bias.data, val=0)
            self.u_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.u_bias.data, val=0)
        else:
            self.z_bias = None
            self.u_bias = None

    def forward(self, graph, inputs, state):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # input to latent z
        z = self.gconv_z(graph, inputs)  # (num_nodes, hidden_dim)
        if self.add_bias:
            z = z + self.z_bias
        z = torch.tanh(z)

        # update gate
        u = self.gconv_u(
            graph, torch.cat([state, z], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            u = u + self.u_bias
        u = torch.sigmoid(u)

        new_state = u * state + (1 - u) * z

        return new_state


class GConvRNNCell(nn.Module):
    """
    Graph Convolution Vanilla RNN Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        rnn_activation_fn="tanh",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        neighbor_sampling=False,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        if rnn_activation_fn == "tanh":
            self.rnn_activation = torch.tanh
        elif rnn_activation_fn == "relu":
            self.rnn_activation = F.relu
        else:
            raise NotImplementedError
        self.rnn_activation_fn = rnn_activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias
        self.neighbor_sampling = neighbor_sampling

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.gconv = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            neighbor_sampling=neighbor_sampling,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.bias.data, val=0)
        else:
            self.bias = None

    def forward(self, graph, inputs, state):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # forget gate
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv(graph, inputs_state)  # (num_nodes, hidden_dim)
        if self.add_bias:
            h = h + self.bias

        new_state = self.rnn_activation(h)

        return new_state


class GConvMemoryGRUCell(nn.Module):
    """
    Graph conv + HiPPO-GRU cell
    """

    name = None
    valid_keys = [
        "uxh",
        "ux",
        "uh",
        "um",
        "hxm",
        "hx",
        "hm",
        "hh",
        "bias",
    ]

    def default_initializers(self):
        return {
            "uxh": "uniform",
            "hxm": "xavier",
            "hx": "xavier",
            "hm": "xavier",
            "um": "zero",
        }

    def default_architecture(self):
        return {
            "ux": True,  # input to memory
            # 'uh': True, # hidden to memory
            "um": False,  # memory to memory
            "hx": True,  # input to hidden
            "hm": True,  # memory to hidden
            "bias": True,
        }

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size,
        memory_order,
        memory_activation="id",
        memory_output=False,
        g_conv="gcn",
        num_gcn_layers=1,
        activation_fn="elu",
        dropout=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.architecture = self.default_architecture()
        self.initializers = self.default_initializers()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_order = memory_order

        self.memory_activation = memory_activation
        self.memory_output = memory_output
        self.g_conv = g_conv
        self.num_gcn_layers = num_gcn_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device

        self.reset_parameters()

        self.input_to_hidden_size = self.input_size if self.architecture["hx"] else 0
        self.input_to_memory_size = self.input_size if self.architecture["ux"] else 0

        # Construct and initialize u --- memory
        self.W_uxh = nn.Linear(
            self.input_to_memory_size + self.hidden_size,
            self.memory_size,
            bias=self.architecture["bias"],
        )

        if "uxh" in self.initializers:
            get_initializer(self.initializers["uxh"], self.memory_activation)(
                self.W_uxh.weight
            )
        if "ux" in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers["ux"], self.memory_activation)(
                self.W_uxh.weight[:, : self.input_size]
            )
        if "uh" in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers["uh"], self.memory_activation)(
                self.W_uxh.weight[:, self.input_size :]
            )

        # Construct and initialize h
        self.memory_to_hidden_size = (
            self.memory_size * self.memory_order if self.architecture["hm"] else 0
        )

        in_size = self.input_to_hidden_size
        candidate_in_size = self.input_to_hidden_size + self.hidden_size
        if self.architecture["hm"]:  # memory to hidden
            in_size += self.memory_to_hidden_size
            candidate_in_size += self.memory_to_hidden_size
        # gconv_gate includes reset and update gates, that's why hidden_dim * 2
        self.gconv_gate = GConvLayers(
            in_dim=in_size,
            hidden_dim=hidden_size * 2,
            num_gcn_layers=num_gcn_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        self.gconv_candidate = GConvLayers(
            in_dim=candidate_in_size,
            hidden_dim=hidden_size,
            num_gcn_layers=num_gcn_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        # note that the biases are initialized as zeros
        if self.architecture["bias"]:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_size * 2,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_size,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

        if self.architecture["um"]:  # default is False
            # No bias here because the implementation is awkward otherwise, but probably doesn't matter
            self.W_um = nn.Parameter(torch.Tensor(self.memory_size, self.memory_order))
            get_initializer(self.initializers["um"], self.memory_activation)(self.W_um)

    def reset_parameters(self):
        # super().reset_parameters()
        self.memory_activation_fn = get_activation(
            self.memory_activation, self.memory_size
        )

    def forward(self, graph, input, state):
        """
        Args:
            graph: DGL graph
            input: shape (num_nodes, input_size)
            state: tuple of (h, m, time_step),
                h shape (num_nodes, hidden_size),
                m shape (num_nodes, memory_size, memory_order)
                time_step: int
        Returns:
            output: current hidden state, shape (num_nodes, hidden_size)
            new_state: tuple of updated (h, m, time_step)
        """
        h, m, time_step = state  # h_t-1, m_t-1, t

        input_to_hidden = (
            input if self.architecture["hx"] else input.new_empty((0,))
        )  # Default is True
        input_to_memory = (
            input if self.architecture["ux"] else input.new_empty((0,))
        )  # Default is True

        # Construct the update features
        memory_preact = self.W_uxh(
            torch.cat((input_to_memory, h), dim=-1)
        )  # (batch, memory_size)
        u = self.memory_activation_fn(memory_preact)  # (batch, memory_size)
        # print('u shape:', u.shape)

        # Update the memory
        m = self.update_memory(m, u, time_step)  # (batch, memory_size, memory_order)
        # print('m updated:', m.shape)

        # Update hidden state from memory
        if self.architecture["hm"]:  # Default is True
            memory_to_hidden = m.view(
                input.shape[0], self.memory_size * self.memory_order
            )
        else:
            memory_to_hidden = input.new_empty((0,))
        m_inputs = torch.cat((input_to_hidden, memory_to_hidden), dim=-1)  # [x_t, m_t]

        r_u = self.gconv_gate(graph, m_inputs)
        if self.architecture["bias"]:
            r_u = r_u + self.gate_bias
        r_u = torch.sigmoid(r_u)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(r_u, split_size_or_sections=self.hidden_size, dim=-1)

        # candidate gate
        c = self.gconv_candidate(
            graph, torch.cat([m_inputs, r * h], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.architecture["bias"]:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        h = u * h + (1.0 - u) * c
        new_state = (h, m, time_step + 1)

        return new_state

    def update_memory(self, m, u, time_step):
        pass
        # raise NotImplementedError

    def default_state(self, input, batch_size=None):
        batch_size = input.size(0) if batch_size is None else batch_size
        return (
            input.new_zeros(batch_size, self.hidden_size, requires_grad=False),
            input.new_zeros(
                batch_size, self.memory_size, self.memory_order, requires_grad=False
            ),
            0,
        )

    def output(self, state):
        """Converts a state into a single output (tensor)"""
        h, m, time_step = state

        if self.memory_output:
            hm = torch.cat(
                (h, m.view(m.shape[0], self.memory_size * self.memory_order)), dim=-1
            )
            return hm
        else:
            return h

    def state_size(self):
        return self.hidden_size + self.memory_size * self.memory_order

    def output_size(self):
        if self.memory_output:
            return self.hidden_size + self.memory_size * self.memory_order
        else:
            return self.hidden_size


class GConvLSICell(GConvMemoryGRUCell):
    """A cell implementing Linear 'Scale' Invariant dynamics: c' = 1/t (Ac + Bf)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size,
        memory_order,
        A,
        B,
        init_t=0,  # 0 for special case at t=0 (new code), else old code without special case
        max_length=1024,
        discretization="bilinear",
        g_conv="gcn",
        num_gcn_layers=1,
        activation_fn="elu",
        dropout=0,
        device="cpu",
        **kwargs,
    ):
        """
        # TODO: make init_t start at arbitrary time (instead of 0 or 1)
        """

        # B should have shape (N, 1)
        assert len(B.shape) == 2 and B.shape[1] == 1

        super(GConvLSICell, self).__init__(
            input_size, hidden_size, memory_size, memory_order, **kwargs
        )

        assert isinstance(init_t, int)
        self.init_t = init_t
        self.max_length = max_length

        A_stacked = np.empty((max_length, memory_order, memory_order), dtype=A.dtype)
        B_stacked = np.empty((max_length, memory_order), dtype=B.dtype)
        B = B[:, 0]
        N = memory_order
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization in forward_aliases:
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization in backward_aliases:
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization in bilinear_aliases:
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, Bt, lower=True
                )
            elif discretization in zoh_aliases:
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(memory_order)  # puts into form: x += Ax
        self.register_buffer("A", torch.Tensor(A_stacked))
        self.register_buffer("B", torch.Tensor(B_stacked))

    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1)  # (B, M, 1)
        t = time_step - 1 + self.init_t
        if t < 0:
            return F.pad(u, (0, self.memory_order - 1))
        else:
            if t >= self.max_length:
                t = self.max_length - 1
            return m + F.linear(m, self.A[t]) + F.linear(u, self.B[t])


class GConvOPLSICell(GConvLSICell):
    measure = None

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size=1,
        memory_order=-1,
        measure_args={},
        **kwargs,
    ):
        if memory_order < 0:
            memory_order = hidden_size

        A, B = transition(type(self).measure, memory_order, **measure_args)
        super(GConvOPLSICell, self).__init__(
            input_size, hidden_size, memory_size, memory_order, A, B, **kwargs
        )


class LegendreScaleGConvCell(GConvOPLSICell):
    name = "legs_gconv"
    measure = "legs"


class GraphRNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        num_gru_layers,
        g_conv="multihead_gat",
        n_classes=1,
        dropout=0.0,
        activation_fn="elu",
        norm=None,
        rnn_activation_fn="tanh",
        device=None,
        is_classifier=True,
        t_model="gru",
        final_pool="last",
        add_bias=True,
        memory_size=1,
        memory_order=-1,
        ehr_encoder_name=None,
        ehr_config=None,
        ehr_checkpoint_path=None,
        freeze_pretrained=False,
        cat_idxs=None,
        cat_dims=None,
        cat_emb_dim=None,
        **kwargs,
    ):
        super(GraphRNN, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.g_conv = g_conv
        self.n_classes = n_classes
        self.activation_fn = activation_fn
        self.rnn_activation_fn = rnn_activation_fn
        self.device = device
        self.is_classifier = is_classifier
        self.t_model = t_model
        self.final_pool = final_pool
        if self.final_pool == "cat":
            self.adapt_cat = AdaptiveConcatPoolRNN()
        self.add_bias = add_bias
        self.memory_size = memory_size
        self.memory_order = memory_order
        self.ehr_encoder_name = ehr_encoder_name
        self.ehr_config = ehr_config
        self.cat_emb_dim = cat_emb_dim

        # ehr encoder
        if ehr_encoder_name is not None:
            if ehr_encoder_name == "tabnet":
                self.ehr_encoder = tab_network.TabNet(
                    input_dim=in_dim,
                    output_dim=n_classes,  # dummy
                    cat_idxs=cat_idxs,
                    cat_dims=cat_dims,
                    **ehr_config,
                )
                if ehr_checkpoint_path is not None:
                    update_state_dict = copy.deepcopy(self.ehr_encoder.state_dict())
                    ckpt = torch.load(ehr_checkpoint_path)

                    for param, weights in ckpt["model_state"].items():
                        if param.startswith("encoder"):
                            # Convert encoder's layers name to match
                            new_param = "tabnet." + param
                        else:
                            new_param = param
                        if self.ehr_encoder.state_dict().get(new_param) is not None:
                            # update only common layers
                            update_state_dict[new_param] = weights
                    self.ehr_encoder.load_state_dict(update_state_dict)
                    print("Loaded pretrained TabNet...")
                    if freeze_pretrained:
                        for param in self.ehr_encoder.parameters():
                            param.requires_grad = False
                    print("Tabnet params frozen...")

                self.in_dim = ehr_config["n_d"]
                in_dim = ehr_config["n_d"]

            elif ehr_encoder_name == "embedder":
                print("Using embedder to embed ehr data...")
                self.embedder = tab_network.EmbeddingGenerator(
                    input_dim=in_dim,
                    cat_dims=cat_dims,
                    cat_idxs=cat_idxs,
                    cat_emb_dim=cat_emb_dim,
                )
                in_dim = (in_dim - len(cat_idxs)) + len(cat_idxs) * cat_emb_dim
            else:
                raise NotImplementedError

        # GConvGRU layers
        self.layers = nn.ModuleList()
        if t_model == "gru":
            self.layers.append(
                GConvGRUCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    norm=norm,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )

            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvGRUCell(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        norm=norm,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        elif t_model == "hippo-legs":
            self.layers.append(
                LegendreScaleGConvCell(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    memory_size=memory_size,
                    memory_order=memory_order,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                )
            )
            for i in range(1, num_gru_layers):
                self.layers.append(
                    LegendreScaleGConvCell(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        memory_size=memory_size,
                        memory_order=memory_order,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        **kwargs,
                    )
                )
        elif t_model == "mgu":
            self.layers.append(
                GConvMGUCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )
            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvMGUCell(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        elif t_model == "rnn":
            self.layers.append(
                GConvRNNCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    rnn_activation_fn=rnn_activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )
            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvRNNCell(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        rnn_activation_fn=rnn_activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        elif t_model == "minimalrnn":
            self.layers.append(
                GConvMinimalRNNCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    rnn_activation_fn=rnn_activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )
            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvMinimalRNNCell(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        rnn_activation_fn=rnn_activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        else:
            raise NotImplementedError

        if is_classifier:
            if self.final_pool == "cat":
                self.fc = nn.Linear(hidden_dim * 3, n_classes)
            else:
                self.fc = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.kwargs = kwargs

    def forward(self, graph, inputs, init_state=None):
        """
        Args:
            graph: list of graphs, if non-time-varying, len(graph) = 1; otherwise len(graph) = max_seq_len
            inputs: input features, shape (num_nodes, seq_len, input_dim), where batch is number of nodes
            init_state: GRU hidden state, shape (num_nodes, gru_dim).
                If None, will initialize init_state
        """
        num_nodes, max_seq_len, in_dim = inputs.shape

        if self.ehr_encoder_name == "tabnet":
            x = self.ehr_encoder.embedder(inputs.view(num_nodes * max_seq_len, -1))
            steps_output, _ = self.ehr_encoder.tabnet.encoder(x)
            inputs = torch.sum(
                torch.stack(steps_output, dim=0), dim=0
            )  # (batch*seq_len, n_d)
            inputs = inputs.reshape(num_nodes, max_seq_len, -1)
        elif self.ehr_encoder_name == "embedder":
            inputs = inputs.reshape(num_nodes * max_seq_len, -1)
            inputs = self.embedder(inputs).reshape(num_nodes, max_seq_len, -1)

        inputs = torch.transpose(
            inputs, dim0=0, dim1=1
        )  # (max_seq_len, num_nodes, in_dim)
        # print("inputs shape:", inputs.shape)

        # initialize GRU hidden states
        if init_state is None:
            hidden_state = self.init_hidden(num_nodes)
        else:
            hidden_state = init_state
        if self.t_model == "hippo-legs":
            memory = self.init_memory(num_nodes)

        # loop over GRU layers
        curr_inputs = inputs
        for idx_gru in range(self.num_gru_layers):
            if self.t_model != "hippo-legs":
                state = hidden_state[idx_gru, :]
            else:
                state = (hidden_state[idx_gru, :], memory[idx_gru, :], 0)

            outputs_inner = []  # inner outputs within GRU layers
            # loop over time
            for t in range(max_seq_len):
                if len(graph) == 1:
                    state = self.layers[idx_gru](
                        graph[0], curr_inputs[t, :, :], state
                    )  # (num_nodes, hidden_dim)
                else:
                    state = self.layers[idx_gru](
                        graph[t], curr_inputs[t, :, :], state
                    )  # (num_nodes, hidden_dim)
                if self.t_model != "hippo-legs":
                    outputs_inner.append(state)
                else:
                    outputs_inner.append(state[0])

            # input to next GRU layer is the previous GRU layer's last hidden state * dropout
            # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            curr_inputs = torch.stack(
                outputs_inner, dim=0
            )  # (seq_len, num_nodes, hidden_dim)
            if idx_gru != (self.num_gru_layers - 1):
                #                 curr_inputs = self.dropout(curr_inputs)
                curr_inputs = self.activation(self.dropout(curr_inputs))
        gru_out = curr_inputs  # (seq_len, num_nodes, hidden_dim)

        if self.final_pool == "last":
            # get last relevant time step output
            out = gru_out[-1, :, :]
        elif self.final_pool == "mean":
            out = torch.mean(gru_out, dim=0)
        elif self.final_pool == "cat":
            gru_out = gru_out.transpose(0, 2)  # (hidden_dim, num_nodes, seq_len)
            gru_out = gru_out.transpose(0, 1)
            out = self.adapt_cat(gru_out)
        else:
            out, _ = torch.max(gru_out, dim=0)

        # Dropout -> ReLU -> FC
        if self.is_classifier:
            # logits = self.fc(self.activation(self.dropout(out))) # (num_nodes,)
            logits = self.fc(self.dropout(out))
            return logits, out
        else:
            return out

    def init_hidden(self, batch_size):
        init_states = []
        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                batch_size, self.hidden_dim, requires_grad=False
            ).to(self.device)
            init_states.append(curr_init)
        init_states = torch.stack(init_states, dim=0).to(
            self.device
        )  # (num_gru_layers, num_nodes, gru_dim)
        return init_states

    def init_memory(self, batch_size):
        init_mem = []
        memory_order = (
            self.hidden_dim if (self.memory_order == -1) else self.memory_order
        )
        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                batch_size, self.memory_size, memory_order, requires_grad=False
            ).to(self.device)
            init_mem.append(curr_init)
        init_mem = torch.stack(init_mem, dim=0).to(
            self.device
        )  # (num_gru_layers, num_nodes, gru_dim)
        return init_mem


class GraphTransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        g_conv="multihead_gat",
        n_classes=1,
        dropout=0.0,
        activation_fn="elu",
        device=None,
        is_classifier=True,
        final_pool="last",
        trans_nhead=8,
        trans_dim_feedforward=128,
        trans_activation="relu",
        att_neighbor=False,
        init_eps=0,
        learn_eps=False,
        aggregator_type="mean",
        **kwargs,
    ):
        super(GraphTransformer, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.g_conv = g_conv
        self.n_classes = n_classes
        self.device = device
        self.is_classifier = is_classifier
        self.final_pool = final_pool
        self.trans_nhead = trans_nhead
        self.trans_dim_feedforward = trans_dim_feedforward
        self.trans_activation = trans_activation
        self.att_neighbor = att_neighbor
        if aggregator_type == "sum":
            self._reducer = dgl.function.sum
        elif aggregator_type == "max":
            self._reducer = dgl.function.max
        elif aggregator_type == "mean":
            self._reducer = dgl.function.mean
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        # Graph conv layers
        self.gconv_layers = GConvLayers(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            is_classifier=False,
            aggregator_type=aggregator_type,
            learn_eps=learn_eps,
            **kwargs,
        )

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

        #         transformer_in = hidden_dim if (not att_neighbor) else (hidden_dim * 2)
        transformer_in = hidden_dim

        # Position embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(num_nodes, seq_len, transformer_in)
        )
        #         self.pos_encoder = PositionalEncoding(d_model=transformer_in, dropout=dropout)

        # Transformer layer
        self.transformer = TransformerEncoderLayer(
            d_model=transformer_in,
            nhead=trans_nhead,
            dim_feedforward=trans_dim_feedforward,
            dropout=dropout,
            activation=trans_activation,
            layer_norm_eps=1e-5,
            device=device,
            dtype=torch.float32,
        )

        if is_classifier:
            self.fc = nn.Linear(transformer_in, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, inputs):
        """
        Args:
            graph: list of graphs, if non-time-varying, len(graph) = 1; otherwise len(graph) = max_seq_len
            inputs: input features, shape (num_nodes, seq_len, input_dim), where batch is number of nodes
        Returns:
            logits: shape (num_nodes, num_classes)
            out: shape (num_nodes, hidden_dim)
        """
        assert len(graph) == 1
        graph = graph[0]

        num_nodes, max_seq_len, in_dim = inputs.shape

        inputs = torch.transpose(
            inputs, dim0=0, dim1=1
        )  # (max_seq_len, num_nodes, in_dim)
        # print("inputs shape:", inputs.shape)

        # pass through graph conv layers
        h = []
        for t in range(max_seq_len):
            curr_h = self.gconv_layers(graph, inputs[t, :])
            h.append(curr_h)
        h = torch.stack(h, dim=1)  # (num_nodes, max_seq_len, hidden_dim)

        # aggregate nodes' neighbors, similar to GIN
        # https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/conv/ginconv.html#GINConv
        if self.att_neighbor:
            with graph.local_scope():
                aggregate_fn = dgl.function.copy_src("h", "m")
                if "weights" in graph.edata.keys():
                    edge_weight = graph.edata["weights"].to(dtype=torch.float32)
                    assert edge_weight.shape[0] == graph.number_of_edges()
                    graph.edata["_edge_weight"] = edge_weight
                    aggregate_fn = dgl.function.u_mul_e("h", "_edge_weight", "m")
                feat_src, feat_dst = expand_as_pair(h, graph)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, self._reducer("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                h = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
        #             h = torch.cat([h, h_neigh], dim=-1) # (num_nodes, max_seq_len, hidden_dim*2)

        # add positional embedding to h
        h += self.pos_embedding
        h = h.transpose(0, 1)  # (max_seq_len, num_nodes, hidden_dim*2)

        # pass through transformer layer
        out = self.transformer(h)  # (max_seq_len, num_nodes, hidden_dim*2)
        #         print('transformer out:',out.shape)

        if self.final_pool == "last":
            out = out[-1, :, :]
        elif self.final_pool == "mean":
            out = torch.mean(out, dim=0)
        else:
            out, _ = torch.max(out, dim=0)

        # Dropout -> FC
        if self.is_classifier:
            logits = self.fc(self.dropout(out))
            return logits, out
        else:
            return out


######## Neighborhood Sampling Code Below ########
class GConvLayers_NeighborSampling(nn.Module):
    """
    Multi-layer GCN/GAT/Multi-head GAT/GraphSAGE/Gated GAT
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        g_conv="graphsage",
        activation_fn="relu",
        dropout=0.0,
        device=None,
        is_classifier=False,
        **kwargs,
    ):
        super(GConvLayers_NeighborSampling, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.g_conv = g_conv
        self.activation_fn = activation_fn
        self.device = device
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.is_classifier = is_classifier

        self.layers = nn.ModuleList()
        if g_conv == "multihead_gat":
            multihead_gat = GAT(
                num_layers=num_gcn_layers,
                in_dim=in_dim,
                num_hidden=hidden_dim,
                num_heads=kwargs["num_heads"],
                activation=None,  # we wil add activation later in forward
                feat_drop=dropout,
                attn_drop=dropout,
                negative_slope=kwargs["negative_slope"],
                residual=kwargs["gat_residual"],
            )
            self.layers = multihead_gat.layers

        elif g_conv == "gat":
            gat = GAT(
                num_layers=num_gcn_layers,
                in_dim=in_dim,
                num_hidden=hidden_dim,
                num_heads=1,
                activation=None,  # we wil add activation later in forward
                feat_drop=dropout,
                attn_drop=dropout,
                negative_slope=kwargs["negative_slope"],
                residual=kwargs["gat_residual"],
            )
            self.layers = gat.layers

        elif g_conv == "graphsage":
            graphsage = GraphSAGE(
                in_feats=in_dim,
                n_hidden=hidden_dim,
                n_layers=num_gcn_layers,
                activation=None,
                dropout=dropout,
                aggregator_type=kwargs["aggregator_type"],
            )
            self.layers = graphsage.layers
        elif g_conv == "gaan":
            self.layers.append(
                GatedGAT(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    map_feats=kwargs["gaan_map_feats"],
                    num_heads=kwargs["num_heads"],
                    activation=None,  # we wil add activation later in forward
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=kwargs["negative_slope"],
                    residual=kwargs["gat_residual"],
                )
            )
            for _ in range(1, num_gcn_layers):
                self.layers.append(
                    GatedGAT(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        map_feats=kwargs["gaan_map_feats"],
                        num_heads=kwargs["num_heads"],
                        activation=None,  # we wil add activation later in forward
                        feat_drop=dropout,
                        attn_drop=dropout,
                        negative_slope=kwargs["negative_slope"],
                        residual=kwargs["gat_residual"],
                    )
                )

        else:
            gcn = GCN(
                in_dim, hidden_dim, num_gcn_layers, activation=None, dropout=dropout
            )
            self.layers = gcn.layers

        # optionally for non-temporal models
        if self.is_classifier:
            self.fc = nn.Linear(hidden_dim, kwargs["num_classes"])

    def forward(self, blocks, inputs):
        """
        Args:
            inputs: shape (batch, in_dim)
        Returns:
            h: shape (batch, hidden_dim) using "mean" aggregate or (batch, hidden_dim*num_heads) using
                "cat" aggregate
        """

        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

            # NEW
            if self.g_conv == "gat" or self.g_conv == "multihead_gat":
                h = h.flatten(1)

            if l != self.num_gcn_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)

        if self.is_classifier:
            logits = self.fc(h)
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return logits, h
        else:
            return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        for l, layer in enumerate(self.layers):
            if self.g_conv != "multihead_gat":
                y = torch.zeros(g.num_nodes(), self.hidden_dim)
            else:
                if l != len(self.layers) - 1:
                    y = torch.zeros(g.num_nodes(), self.hidden_dim * self.num_heads)
                else:
                    y = torch.zeros(g.num_nodes(), self.hidden_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).type(torch.int32).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes.long()].to(device)
                h = layer(block, h)
                if self.g_conv == "gat" or self.g_conv == "multihead_gat":
                    h = h.flatten(1)
                if l != self.num_gcn_layers - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes.long()] = h.cpu()

            x = y

        if self.is_classifier:
            logits = self.fc(y.to(device))
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return logits, y
        else:
            return y.to(device)


class GConvRNNCell_NeighborSampling(nn.Module):
    """
    Graph Convolution Vanilla RNN Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        rnn_activation_fn="tanh",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvRNNCell_NeighborSampling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        if rnn_activation_fn == "tanh":
            self.rnn_activation = torch.tanh
        elif rnn_activation_fn == "relu":
            self.rnn_activation = F.relu
        else:
            raise NotImplementedError
        self.rnn_activation_fn = rnn_activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.gconv = GConvLayers_NeighborSampling(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.bias.data, val=0)
        else:
            self.bias = None

    def forward(self, blocks, inputs, state, output_nodes):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv(blocks, inputs_state)  # (num_nodes, hidden_dim)
        if self.add_bias:
            h = h + self.bias

        new_state = state
        # NOTE, we only update the state for output nodes in the bipartite graph
        new_state[output_nodes.long()] = self.rnn_activation(h)

        return new_state

    def inference(
        self, graph, inputs, state, device="cpu", batch_size=256, num_workers=8
    ):
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv.inference(
            graph,
            inputs_state,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if self.add_bias:
            h = h + self.bias

        new_state = self.rnn_activation(h)

        return new_state


class GConvGRUCell_NeighborSampling(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="gat",
        num_gconv_layers=1,
        activation_fn="elu",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs,
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvGRUCell_NeighborSampling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        # gconv_gate includes reset and update gates, that's why hidden_dim * 2
        self.gconv_gate = GConvLayers_NeighborSampling(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        self.gconv_candidate = GConvLayers_NeighborSampling(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim * 2,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

    def forward(self, blocks, inputs, state, output_nodes):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # reset and update gates
        # NOTE: we make non-output nodes reset gate & update gate & candidate to be identity
        r = torch.ones(state.shape).to(inputs.device)
        u = torch.ones(state.shape).to(inputs.device)
        c = torch.ones(state.shape).to(inputs.device)

        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv_gate(blocks, inputs_state)  # (num_nodes, hidden_dim*2)
        if self.add_bias:
            h = h + self.gate_bias
        h = torch.sigmoid(h)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r_out, u_out = torch.split(h, split_size_or_sections=self.hidden_dim, dim=-1)
        r[output_nodes] = r_out
        u[output_nodes] = u_out

        # candidate
        c_out = self.gconv_candidate(
            blocks, torch.cat([inputs, r * state], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            c_out = c_out + self.candidate_bias
        c_out = torch.tanh(c_out)
        c[output_nodes] = c_out

        new_state = u * state + (1 - u) * c

        return new_state

    def inference(
        self, graph, inputs, state, device="cpu", batch_size=256, num_workers=8
    ):
        """
        NOTE: This will involve 3 x seq_len x num_gnn_layers looping over the nodes
        Not elegant but perhaps ok
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # reset and update gates
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv_gate.inference(
            graph,
            inputs_state,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ).to(
            device
        )  # (num_nodes, hidden_dim*2)
        if self.add_bias:
            h = h + self.gate_bias
        h = torch.sigmoid(h)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(h, split_size_or_sections=self.hidden_dim, dim=-1)

        # candidate
        c = self.gconv_candidate.inference(
            graph,
            torch.cat([inputs, r * state], dim=-1),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ).to(
            device
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        new_state = u * state + (1 - u) * c

        return new_state


class GConvMemoryGRUCell_NeighborSampling(nn.Module):
    """
    Graph conv + HiPPO-GRU cell
    """

    name = None
    valid_keys = [
        "uxh",
        "ux",
        "uh",
        "um",
        "hxm",
        "hx",
        "hm",
        "hh",
        "bias",
    ]

    def default_initializers(self):
        return {
            "uxh": "uniform",
            "hxm": "xavier",
            "hx": "xavier",
            "hm": "xavier",
            "um": "zero",
        }

    def default_architecture(self):
        return {
            "ux": True,  # input to memory
            # 'uh': True, # hidden to memory
            "um": False,  # memory to memory
            "hx": True,  # input to hidden
            "hm": True,  # memory to hidden
            "bias": True,
        }

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size,
        memory_order,
        memory_activation="id",
        memory_output=False,
        g_conv="gcn",
        num_gcn_layers=1,
        activation_fn="elu",
        dropout=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.architecture = self.default_architecture()
        self.initializers = self.default_initializers()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_order = memory_order

        self.memory_activation = memory_activation
        self.memory_output = memory_output
        self.g_conv = g_conv
        self.num_gcn_layers = num_gcn_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device

        self.reset_parameters()

        self.input_to_hidden_size = self.input_size if self.architecture["hx"] else 0
        self.input_to_memory_size = self.input_size if self.architecture["ux"] else 0

        # Construct and initialize u --- memory
        self.W_uxh = nn.Linear(
            self.input_to_memory_size + self.hidden_size,
            self.memory_size,
            bias=self.architecture["bias"],
        )

        if "uxh" in self.initializers:
            get_initializer(self.initializers["uxh"], self.memory_activation)(
                self.W_uxh.weight
            )
        if "ux" in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers["ux"], self.memory_activation)(
                self.W_uxh.weight[:, : self.input_size]
            )
        if "uh" in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers["uh"], self.memory_activation)(
                self.W_uxh.weight[:, self.input_size :]
            )

        # Construct and initialize h
        self.memory_to_hidden_size = (
            self.memory_size * self.memory_order if self.architecture["hm"] else 0
        )

        in_size = self.input_to_hidden_size
        candidate_in_size = self.input_to_hidden_size + self.hidden_size
        if self.architecture["hm"]:  # memory to hidden
            in_size += self.memory_to_hidden_size
            candidate_in_size += self.memory_to_hidden_size
        print("candidate_in_size in size:", candidate_in_size)
        # gconv_gate includes reset and update gates, that's why hidden_dim * 2
        self.gconv_gate = GConvLayers_NeighborSampling(
            in_dim=in_size,
            hidden_dim=hidden_size * 2,
            num_gcn_layers=num_gcn_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        self.gconv_candidate = GConvLayers_NeighborSampling(
            in_dim=candidate_in_size,
            hidden_dim=hidden_size,
            num_gcn_layers=num_gcn_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        # note that the biases are initialized as zeros
        if self.architecture["bias"]:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_size * 2,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_size,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

        if self.architecture["um"]:  # default is False
            # No bias here because the implementation is awkward otherwise, but probably doesn't matter
            self.W_um = nn.Parameter(torch.Tensor(self.memory_size, self.memory_order))
            get_initializer(self.initializers["um"], self.memory_activation)(self.W_um)

    def reset_parameters(self):
        # super().reset_parameters()
        self.memory_activation_fn = get_activation(
            self.memory_activation, self.memory_size
        )

    def forward(self, blocks, input, state, output_nodes):
        """
        Args:
            blocks: bipartite graph block
            input: shape (num_nodes, input_size)
            state: tuple of (h, m, time_step),
                h shape (num_nodes, hidden_size),
                m shape (num_nodes, memory_size, memory_order)
                time_step: int
            output_nodes: output nodes of the bipartite graph
        Returns:
            new_state: tuple of updated (h, m, time_step)
        """
        h, m, time_step = state  # h_t-1, m_t-1, t

        # NOTE: we make non-output nodes reset gate & update gate & candidate to be identity
        reset = torch.ones(h.shape).to(input.device)
        update = torch.ones(h.shape).to(input.device)
        candidate = torch.ones(h.shape).to(input.device)

        input_to_hidden = (
            input if self.architecture["hx"] else input.new_empty((0,))
        )  # Default is True
        input_to_memory = (
            input if self.architecture["ux"] else input.new_empty((0,))
        )  # Default is True

        # Construct the update features
        memory_preact = self.W_uxh(
            torch.cat((input_to_memory, h), dim=-1)
        )  # (batch, memory_size)
        u = self.memory_activation_fn(memory_preact)  # (batch, memory_size)
        # print('u shape:', u.shape)

        # Update the memory
        m = self.update_memory(m, u, time_step)  # (batch, memory_size, memory_order)
        # print('m updated:', m.shape)

        # Update hidden state from memory
        if self.architecture["hm"]:  # Default is True
            memory_to_hidden = m.view(
                input.shape[0], self.memory_size * self.memory_order
            )
        else:
            memory_to_hidden = input.new_empty((0,))
        m_inputs = torch.cat((input_to_hidden, memory_to_hidden), dim=-1)  # [x_t, m_t]

        r_u = self.gconv_gate(blocks, m_inputs)
        if self.architecture["bias"]:
            r_u = r_u + self.gate_bias
        r_u = torch.sigmoid(r_u)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(r_u, split_size_or_sections=self.hidden_size, dim=-1)
        reset[output_nodes] = r
        update[output_nodes] = u

        # candidate gate
        c = self.gconv_candidate(
            blocks, torch.cat([m_inputs, reset * h], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.architecture["bias"]:
            c = c + self.candidate_bias
        c = torch.tanh(c)
        candidate[output_nodes] = c

        h = update * h + (1.0 - update) * candidate
        new_state = (h, m, time_step + 1)

        return new_state

    def inference(
        self, graph, input, state, device="cpu", batch_size=256, num_workers=0
    ):
        """
        Args:
            graph: DGL graph
            input: shape (num_nodes, input_size)
            state: tuple of (h, m, time_step),
                h shape (num_nodes, hidden_size),
                m shape (num_nodes, memory_size, memory_order)
                time_step: int
        Returns:
            output: current hidden state, shape (num_nodes, hidden_size)
            new_state: tuple of updated (h, m, time_step)
        """
        h, m, time_step = state  # h_t-1, m_t-1, t

        input_to_hidden = (
            input if self.architecture["hx"] else input.new_empty((0,))
        )  # Default is True
        input_to_memory = (
            input if self.architecture["ux"] else input.new_empty((0,))
        )  # Default is True

        # Construct the update features
        memory_preact = self.W_uxh(
            torch.cat((input_to_memory, h), dim=-1)
        )  # (batch, memory_size)
        u = self.memory_activation_fn(memory_preact)  # (batch, memory_size)
        # print('u shape:', u.shape)

        # Update the memory
        m = self.update_memory(m, u, time_step)  # (batch, memory_size, memory_order)
        # print('m updated:', m.shape)

        # Update hidden state from memory
        if self.architecture["hm"]:  # Default is True
            memory_to_hidden = m.view(
                input.shape[0], self.memory_size * self.memory_order
            )
        else:
            memory_to_hidden = input.new_empty((0,))
        m_inputs = torch.cat((input_to_hidden, memory_to_hidden), dim=-1)  # [x_t, m_t]

        r_u = self.gconv_gate.inference(
            graph,
            m_inputs,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ).to(
            device
        )  # (num_nodes, hidden_dim*2)
        if self.architecture["bias"]:
            r_u = r_u + self.gate_bias
        r_u = torch.sigmoid(r_u)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(r_u, split_size_or_sections=self.hidden_size, dim=-1)

        # candidate gate
        c = self.gconv_candidate.inference(
            graph,
            torch.cat([m_inputs, r * h], dim=-1),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ).to(
            device
        )  # (num_nodes, hidden_dim)
        if self.architecture["bias"]:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        h = u * h + (1.0 - u) * c
        new_state = (h, m, time_step + 1)

        return new_state

    def update_memory(self, m, u, time_step):
        pass
        # raise NotImplementedError

    def default_state(self, input, batch_size=None):
        batch_size = input.size(0) if batch_size is None else batch_size
        return (
            input.new_zeros(batch_size, self.hidden_size, requires_grad=False),
            input.new_zeros(
                batch_size, self.memory_size, self.memory_order, requires_grad=False
            ),
            0,
        )

    def output(self, state):
        """Converts a state into a single output (tensor)"""
        h, m, time_step = state

        if self.memory_output:
            hm = torch.cat(
                (h, m.view(m.shape[0], self.memory_size * self.memory_order)), dim=-1
            )
            return hm
        else:
            return h

    def state_size(self):
        return self.hidden_size + self.memory_size * self.memory_order

    def output_size(self):
        if self.memory_output:
            return self.hidden_size + self.memory_size * self.memory_order
        else:
            return self.hidden_size


class GConvLSICell_NeighborSampling(GConvMemoryGRUCell_NeighborSampling):
    """A cell implementing Linear 'Scale' Invariant dynamics: c' = 1/t (Ac + Bf)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size,
        memory_order,
        A,
        B,
        init_t=0,  # 0 for special case at t=0 (new code), else old code without special case
        max_length=1024,
        discretization="bilinear",
        **kwargs,
    ):
        """
        # TODO: make init_t start at arbitrary time (instead of 0 or 1)
        """

        # B should have shape (N, 1)
        assert len(B.shape) == 2 and B.shape[1] == 1

        super(GConvLSICell_NeighborSampling, self).__init__(
            input_size, hidden_size, memory_size, memory_order, **kwargs
        )

        assert isinstance(init_t, int)
        self.init_t = init_t
        self.max_length = max_length

        A_stacked = np.empty((max_length, memory_order, memory_order), dtype=A.dtype)
        B_stacked = np.empty((max_length, memory_order), dtype=B.dtype)
        B = B[:, 0]
        N = memory_order
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization in forward_aliases:
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization in backward_aliases:
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization in bilinear_aliases:
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, Bt, lower=True
                )
            elif discretization in zoh_aliases:
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(memory_order)  # puts into form: x += Ax
        self.register_buffer("A", torch.Tensor(A_stacked))
        self.register_buffer("B", torch.Tensor(B_stacked))

    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1)  # (B, M, 1)
        t = time_step - 1 + self.init_t
        if t < 0:
            return F.pad(u, (0, self.memory_order - 1))
        else:
            if t >= self.max_length:
                t = self.max_length - 1
            return m + F.linear(m, self.A[t]) + F.linear(u, self.B[t])


class GConvOPLSICell_NeighborSampling(GConvLSICell_NeighborSampling):
    measure = None

    def __init__(
        self,
        input_size,
        hidden_size,
        memory_size=1,
        memory_order=-1,
        measure_args={},
        **kwargs,
    ):
        if memory_order < 0:
            memory_order = hidden_size

        A, B = transition(type(self).measure, memory_order, **measure_args)
        super(GConvOPLSICell_NeighborSampling, self).__init__(
            input_size, hidden_size, memory_size, memory_order, A, B, **kwargs
        )


class LegendreScaleGConvCell_NeighborSampling(GConvOPLSICell_NeighborSampling):
    name = "legs_gconv_neighbor_sampling"
    measure = "legs"


class GraphRNN_NeighborSampling(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        num_gru_layers,
        g_conv="multihead_gat",
        n_classes=1,
        dropout=0.0,
        activation_fn="elu",
        rnn_activation_fn="tanh",
        device=None,
        is_classifier=True,
        t_model="gru",
        final_pool="last",
        add_bias=True,
        memory_size=1,
        memory_order=-1,
        **kwargs,
    ):
        super(GraphRNN_NeighborSampling, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.g_conv = g_conv
        self.n_classes = n_classes
        self.activation_fn = activation_fn
        self.rnn_activation_fn = rnn_activation_fn
        self.device = device
        self.is_classifier = is_classifier
        self.t_model = t_model
        self.final_pool = final_pool
        self.add_bias = add_bias
        self.memory_size = memory_size
        self.memory_order = memory_order

        # GConvGRU layers
        self.layers = nn.ModuleList()
        if t_model == "gru":
            self.layers.append(
                GConvGRUCell_NeighborSampling(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )

            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvGRUCell_NeighborSampling(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        elif t_model == "hippo-legs":
            self.layers.append(
                LegendreScaleGConvCell_NeighborSampling(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    memory_size=memory_size,
                    memory_order=memory_order,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                )
            )
            for i in range(1, num_gru_layers):
                self.layers.append(
                    LegendreScaleGConvCell_NeighborSampling(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        memory_size=memory_size,
                        memory_order=memory_order,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        **kwargs,
                    )
                )
        elif t_model == "rnn":
            self.layers.append(
                GConvRNNCell_NeighborSampling(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs,
                )
            )

            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvRNNCell_NeighborSampling(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs,
                    )
                )
        else:
            raise NotImplementedError

        if is_classifier:
            self.fc = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.kwargs = kwargs

    def forward(self, blocks, inputs, output_nodes, init_state=None):
        """
        Args:
            graph: list of graphs, if non-time-varying, len(graph) = 1; otherwise len(graph) = max_seq_len
            inputs: input features, shape (num_nodes, seq_len, input_dim), where batch is number of nodes
            init_state: GRU hidden state, shape (num_nodes, gru_dim).
                If None, will initialize init_state
        """
        num_nodes, max_seq_len, in_dim = inputs.shape

        inputs = torch.transpose(
            inputs, dim0=0, dim1=1
        )  # (max_seq_len, num_nodes, in_dim)

        # initialize GRU hidden states
        if init_state is None:
            hidden_state = self.init_hidden(blocks)
            if self.t_model == "hippo-legs":
                memory = self.init_memory(blocks)
        else:
            hidden_state = init_state

        # loop over GRU layers
        curr_inputs = inputs
        for idx_gru in range(self.num_gru_layers):
            if self.t_model != "hippo-legs":
                state = hidden_state[idx_gru]
            else:
                state = (hidden_state[idx_gru], memory[idx_gru], 0)

            outputs_inner = []  # inner outputs within GRU layers
            # loop over time
            for t in range(max_seq_len):
                state = self.layers[idx_gru](
                    blocks, curr_inputs[t, :, :], state, output_nodes
                )  # (num_nodes, hidden_dim)
                if self.t_model != "hippo-legs":
                    outputs_inner.append(state)
                else:
                    outputs_inner.append(state[0])

            # input to next GRU layer is the previous GRU layer's last hidden state * dropout
            # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            curr_inputs = torch.stack(
                outputs_inner, dim=0
            )  # (seq_len, num_nodes, hidden_dim)
            if idx_gru != (self.num_gru_layers - 1):
                #                 curr_inputs = self.dropout(curr_inputs)
                curr_inputs = self.activation(self.dropout(curr_inputs))
        gru_out = curr_inputs  # (seq_len, num_nodes, hidden_dim)

        if self.final_pool == "last":
            # get last relevant time step output
            out = gru_out[-1, :, :]
        elif self.final_pool == "mean":
            out = torch.mean(gru_out, dim=0)
        else:
            out, _ = torch.max(gru_out, dim=0)

        # Dropout -> ReLU -> FC
        if self.is_classifier:
            logits = self.fc(self.dropout(out))
            return logits, out
        else:
            return out

    def inference(
        self, graph, inputs, init_state=None, device="cpu", batch_size=32, num_workers=0
    ):
        num_nodes, max_seq_len, in_dim = inputs.shape

        inputs = torch.transpose(
            inputs, dim0=0, dim1=1
        )  # (max_seq_len, num_nodes, in_dim)

        # initialize GRU hidden states
        if init_state is None:
            hidden_state = self.init_hidden_inference(num_nodes)
            if self.t_model == "hippo-legs":
                memory = self.init_memory_inference(num_nodes)
        else:
            hidden_state = init_state

        # loop over GRU layers
        curr_inputs = inputs
        for idx_gru in range(self.num_gru_layers):
            if self.t_model != "hippo-legs":
                state = hidden_state[idx_gru]
            else:
                state = (hidden_state[idx_gru], memory[idx_gru], 0)

            outputs_inner = []  # inner outputs within GRU layers
            # loop over time
            for t in range(max_seq_len):
                state = self.layers[idx_gru].inference(
                    graph,
                    curr_inputs[t, :, :],
                    state,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )  # (num_nodes, hidden_dim)
                if self.t_model != "hippo-legs":
                    outputs_inner.append(state)
                else:
                    outputs_inner.append(state[0])

            # input to next GRU layer is the previous GRU layer's last hidden state * dropout
            # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            curr_inputs = torch.stack(
                outputs_inner, dim=0
            )  # (seq_len, num_nodes, hidden_dim)
            if idx_gru != (self.num_gru_layers - 1):
                #                 curr_inputs = self.dropout(curr_inputs)
                curr_inputs = self.activation(self.dropout(curr_inputs))
        gru_out = curr_inputs  # (seq_len, num_nodes, hidden_dim)

        if self.final_pool == "last":
            # get last relevant time step output
            out = gru_out[-1, :, :]
        elif self.final_pool == "mean":
            out = torch.mean(gru_out, dim=0)
        else:
            out, _ = torch.max(gru_out, dim=0)

        # Dropout -> ReLU -> FC
        if self.is_classifier:
            logits = self.fc(self.dropout(out))
            return logits, out
        else:
            return out

    def init_hidden(self, blocks):
        init_states = []

        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                blocks[0].num_src_nodes(), self.hidden_dim, requires_grad=False
            ).to(self.device)
            init_states.append(curr_init)
        return init_states

    def init_memory(self, blocks):
        init_mem = []
        memory_order = (
            self.hidden_dim if (self.memory_order == -1) else self.memory_order
        )
        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                blocks[0].num_src_nodes(),
                self.memory_size,
                memory_order,
                requires_grad=False,
            ).to(self.device)
            init_mem.append(curr_init)
        return init_mem

    def init_hidden_inference(self, num_nodes):
        init_states = []

        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(num_nodes, self.hidden_dim, requires_grad=False).to(
                self.device
            )
            init_states.append(curr_init)
        return init_states

    def init_memory_inference(self, num_nodes):
        init_mem = []
        memory_order = (
            self.hidden_dim if (self.memory_order == -1) else self.memory_order
        )
        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                num_nodes, self.memory_size, memory_order, requires_grad=False
            ).to(self.device)
            init_mem.append(curr_init)
        return init_mem
