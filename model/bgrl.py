"""
Code adapted from https://github.com/Namkyeong/BGRL_Pytorch/blob/main/models.py
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from functools import wraps
import copy
import sys
sys.path.append('../')
from model.model import GConvLayers, GraphRNN


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BGRL(nn.Module):
    """
    Adapted from https://github.com/Namkyeong/BGRL_Pytorch/blob/main/models.py
    """
    def __init__(self, model_name, in_dim, hidden_dim, num_gcn_layers, g_conv, pred_hid, num_classes=3,
                 num_rnn_layers=2, num_heads=3, activation_fn='elu', dropout=0.0, moving_average_decay=0.99, 
                 epochs=1000, device=None, t_model='gru', is_classifier=False, **kwargs):
        super().__init__()
        if model_name == 'stgcn':
            self.student_encoder = GraphRNN(in_dim=in_dim, 
                                hidden_dim=hidden_dim, 
                                num_gcn_layers=num_gcn_layers, 
                                num_gru_layers=num_rnn_layers,
                                n_classes=num_classes,
                                g_conv=g_conv, 
                                dropout=dropout, 
                                activation_fn=activation_fn,
                                device=device,
                                add_bias=True,
                                num_heads=num_heads,
                                is_classifier=is_classifier,
                                **kwargs)
        else:
            self.student_encoder = GConvLayers(in_dim=in_dim, 
                                        hidden_dim=hidden_dim, 
                                        num_gcn_layers=num_gcn_layers,
                                        num_classes=num_classes,
                                        g_conv=g_conv, 
                                        activation_fn=activation_fn,
                                        dropout=dropout,
                                        device=device,
                                        num_heads=num_heads,
                                        is_classifier=is_classifier,
                                        **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.student_predictor = nn.Sequential(nn.Linear(hidden_dim, pred_hid), 
                                               nn.BatchNorm1d(pred_hid, momentum = 0.01), 
                                               nn.PReLU(), 
                                               nn.Linear(pred_hid, hidden_dim))
        self.student_predictor.apply(init_weights)
        self.model_name = model_name
        self.is_classifier = is_classifier
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, graph_v1, graph_v2, graph_orig):
        if self.model_name != 'stgcn':
            feat_v1 = graph_v1.ndata['feat'][:, -1, :]
            feat_v2 = graph_v2.ndata['feat'][:, -1, :]
            if self.is_classifier:
                _, v1_student = self.student_encoder(graph_v1, feat_v1)
                _, v2_student = self.student_encoder(graph_v2, feat_v2)
            else:
                v1_student = self.student_encoder(graph_v1, feat_v1)
                v2_student = self.student_encoder(graph_v2, feat_v2)
        else:
            feat_v1 = graph_v1.ndata['feat']
            feat_v2 = graph_v2.ndata['feat']
            seq_len = graph_v1.ndata['seq_lengths'] # same for both views
            if self.is_classifier:
                _, v1_student = self.student_encoder([graph_v1], feat_v1, seq_len)
                _, v2_student = self.student_encoder([graph_v2], feat_v2, seq_len)
            else:
                v1_student = self.student_encoder([graph_v1], feat_v1, seq_len)
                v2_student = self.student_encoder([graph_v2], feat_v2, seq_len)
        

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            if self.model_name != 'stgcn':
                if self.is_classifier:
                    _, v1_teacher = self.teacher_encoder(graph_v1, feat_v1)
                    _, v2_teacher = self.teacher_encoder(graph_v2, feat_v2)
                else:
                    v1_teacher = self.teacher_encoder(graph_v1, feat_v1)
                    v2_teacher = self.teacher_encoder(graph_v2, feat_v2)
            else:
                seq_len = graph_v1.ndata['seq_lengths']
                if self.is_classifier:
                    _, v1_teacher = self.teacher_encoder([graph_v1], feat_v1, seq_len)
                    _, v2_teacher = self.teacher_encoder([graph_v2], feat_v2, seq_len)
                else:
                    v1_teacher = self.teacher_encoder([graph_v1], feat_v1, seq_len)
                    v2_teacher = self.teacher_encoder([graph_v2], feat_v2, seq_len)
                
        ## self-supervised loss
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())
        loss = loss1 + loss2

        ## supervised
        if self.is_classifier:
            feat_orig = graph_orig.ndata['feat']
            if self.model_name != 'stgcn':
                feat_orig = feat_orig[:, -1, :]
                logits, _ = self.student_encoder(graph_orig, feat_orig)
            else:
                seq_len = graph_orig.ndata['seq_lengths']
                logits, _ = self.student_encoder([graph_orig], feat_orig, seq_len)
            if logits.shape[-1] == 1:
                logits = logits.view(-1)
        else:
            logits = None

        return v1_student, v2_student, loss.mean(), logits
    
class BGRL_FineTune(nn.Module):
    def __init__(self, bgrl_model, emb_dim, num_classes, finetune_linear=False, model_name='stgcn',
                dropout=0.):
        super().__init__()
        assert not(bgrl_model.is_classifier)
        self.bgrl_model = bgrl_model
        self.fc = nn.Linear(emb_dim, num_classes)
        
        set_requires_grad(self.bgrl_model.teacher_encoder, False)
        set_requires_grad(self.bgrl_model.student_predictor, False)
        if finetune_linear:
            set_requires_grad(self.bgrl_model.student_encoder, False)
            
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph, features, seq_lengths=None):
        if self.model_name == 'stgcn':
            emb = self.bgrl_model.student_encoder(graph, features, seq_lengths)
        else:
            emb = self.bgrl_model.student_encoder(graph, features)
        logits = self.fc(self.dropout(emb))
        return logits, emb
    
    
        