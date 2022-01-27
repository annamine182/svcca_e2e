# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class BaseAttention(nn.Module):
    """Base class for attention layers."""

    def __init__(self, query_dim, value_dim, embed_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        pass

    def forward(self, query, value, key_padding_mask=None, state=None):
        # query: bsz x q_hidden
        # value: len x bsz x v_hidden
        # key_padding_mask: len x bsz
        raise NotImplementedError


class BahdanauAttention(BaseAttention):
    """ Bahdanau Attention."""

    def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v = Parameter(torch.Tensor(embed_dim))
        self.normalize = normalize
        if self.normalize:
            self.b = Parameter(torch.Tensor(embed_dim))
            self.g = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.query_proj.weight.data.uniform_(-0.1, 0.1)
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)
        if self.normalize:
            nn.init.constant_(self.b, 0.)
            nn.init.constant_(self.g, math.sqrt(1. / self.embed_dim))

    def forward(self, query, value, key_padding_mask=None, state=None):
        # projected_query: 1 x bsz x embed_dim
        projected_query = self.query_proj(query).unsqueeze(0)
        key = self.value_proj(value)  # len x bsz x embed_dim
        if self.normalize:
            # normed_v = g * v / ||v||
            normed_v = self.g * self.v / torch.norm(self.v)
            attn_scores = (
                normed_v * torch.tanh(projected_query + key + self.b)
            ).sum(dim=2)  # len x bsz
        else:
            attn_scores = self.v * torch.tanh(projected_query + key).sum(dim=2)

        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted value. context: bsz x value_dim
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores

        return context, attn_scores, next_state


class LuongAttention(BaseAttention):
    """ Luong Attention."""

    def __init__(self, query_dim, value_dim, embed_dim=None, scale=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.value_proj = nn.Linear(self.value_dim, self.query_dim, bias=False)
        self.scale = scale
        if self.scale:
            self.g = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        if self.scale:
            nn.init.constant_(self.g, 1.)

    def forward(self, query, value, key_padding_mask=None, state=None):
        query = query.unsqueeze(1)  # bsz x 1 x query_dim
        key = self.value_proj(value).transpose(0, 1)  # bsz x len x query_dim
        attn_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        attn_scores = attn_scores.transpose(0, 1)  # len x bsz
        if self.scale:
            attn_scores = self.g * attn_scores

        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted value. context: bsz x value_dim
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores

        return context, attn_scores, next_state

class BahdanauAttentionMOMA(BaseAttention):
    """ Bahdanau Attention MOMA."""

    def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.query_proj1 = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj1 = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v1 = Parameter(torch.Tensor(embed_dim))

        self.query_proj2 = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj2 = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v2 = Parameter(torch.Tensor(embed_dim))

        self.query_proj3 = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj3 = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v3 = Parameter(torch.Tensor(embed_dim))

        self.gamma1 = nn.Parameter(torch.rand(1))
        self.gamma2 = nn.Parameter(torch.rand(1))
        self.gamma3 = nn.Parameter(torch.rand(1))

        self.normalize = normalize
        if self.normalize:
            self.b1 = Parameter(torch.Tensor(embed_dim))
            self.g1 = Parameter(torch.Tensor(1))

            self.b2 = Parameter(torch.Tensor(embed_dim))
            self.g2 = Parameter(torch.Tensor(1))

            self.b3 = Parameter(torch.Tensor(embed_dim))
            self.g3 = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.query_proj1.weight.data.uniform_(-0.1, 0.1)
        self.value_proj1.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v1, -0.1, 0.1)

        self.query_proj2.weight.data.uniform_(-0.1, 0.1)
        self.value_proj2.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v2, -0.1, 0.1)

        self.query_proj3.weight.data.uniform_(-0.1, 0.1)
        self.value_proj3.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v3, -0.1, 0.1)

        if self.normalize:
            nn.init.constant_(self.b1, 0.)
            nn.init.constant_(self.g1, math.sqrt(1. / self.embed_dim))

            nn.init.constant_(self.b2, 0.)
            nn.init.constant_(self.g2, math.sqrt(1. / self.embed_dim))

            nn.init.constant_(self.b3, 0.)
            nn.init.constant_(self.g3, math.sqrt(1. / self.embed_dim))

    def forward(self, query, value, key_padding_mask=None, state=None):
        #####1

        # projected_query: 1 x bsz x embed_dim
        projected_query1 = self.query_proj1(query).unsqueeze(0)
        key1 = self.value_proj1(value)  # len x bsz x embed_dim
        # print("Block 1 :: attention step 1: projected query {} key {}".format(projected_query1.size(),key1.size()))
        if self.normalize:
            # normed_v = g * v / ||v||
            normed_v1 = self.g1 * self.v1 / torch.norm(self.v1)
            attn_scores1 = (
                normed_v1 * torch.tanh(projected_query1 + key1 + self.b1)
            ).sum(dim=2)  # len x bsz
        else:
            attn_scores1 = self.v1 * torch.tanh(projected_query1 + key1).sum(dim=2)

        # print("Block 1 :: attention step 2: attention energy {}".format(attn_scores1.size()))

        if key_padding_mask is not None:
            attn_scores1 = attn_scores1.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores1)  # FP16 support: cast to float and back

        attn_scores1 = F.softmax(attn_scores1, dim=0)  # srclen x bsz

        # print("Block 1 :: attention step 3: attention score (if padding necessary) {}".format(attn_scores1.size()))


        ######2

        # projected_query: 1 x bsz x embed_dim
        projected_query2 = self.query_proj2(query).unsqueeze(0)
        key2 = self.value_proj2(value)  # len x bsz x embed_dim
        # print("Block 2 :: attention step 1: projected query {} key {}".format(projected_query2.size(), key2.size()))
        if self.normalize:
            # normed_v = g * v / ||v||
            normed_v2 = self.g2 * self.v2 / torch.norm(self.v2)
            attn_scores2 = (
                    normed_v2 * torch.tanh(projected_query2 + key2 + self.b2)
            ).sum(dim=2)  # len x bsz
        else:
            attn_scores2 = self.v2 * torch.tanh(projected_query2 + key2).sum(dim=2)

        # print("Block 2 :: attention step 2: attention energy {}".format(attn_scores2.size()))

        if key_padding_mask is not None:
            attn_scores2 = attn_scores2.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores2)  # FP16 support: cast to float and back

        attn_scores2 = F.softmax(attn_scores2, dim=0)  # srclen x bsz

        # print("Block 2 :: attention step 3: attention score (if padding necessary) {}".format(attn_scores2.size()))

        #####3

        # projected_query: 1 x bsz x embed_dim
        projected_query3 = self.query_proj3(query).unsqueeze(0)
        key3 = self.value_proj3(value)  # len x bsz x embed_dim
        # print("Block 3 :: attention step 1: projected query {} key {}".format(projected_query3.size(), key3.size()))
        if self.normalize:
            # normed_v = g * v / ||v||
            normed_v3 = self.g3 * self.v3 / torch.norm(self.v3)
            attn_scores3 = (
                    normed_v3 * torch.tanh(projected_query3 + key3 + self.b3)
            ).sum(dim=2)  # len x bsz
        else:
            attn_scores3 = self.v3 * torch.tanh(projected_query3 + key3).sum(dim=2)

        # print("Block 3 :: attention step 2: attention energy {}".format(attn_scores3.size()))

        if key_padding_mask is not None:
            attn_scores3 = attn_scores3.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores3)  # FP16 support: cast to float and back

        attn_scores3 = F.softmax(attn_scores3, dim=0)  # srclen x bsz

        # print("Block 3 :: attention step 3: attention score (if padding necessary) {}".format(attn_scores3.size()))

        # sum weighted value. context: bsz x value_dim
        context = self.gamma1*(attn_scores1.unsqueeze(2) * value).sum(dim=0) + \
                  self.gamma2*(attn_scores2.unsqueeze(2) * value).sum(dim=0) +\
                  self.gamma3*(attn_scores3.unsqueeze(2) * value).sum(dim=0)

        attn_scores = self.gamma1*attn_scores1 + self.gamma2*attn_scores2 + self.gamma3*attn_scores3
        next_state = attn_scores

        # print("Final Attention: context {} next state {}".format(context.size(),next_state.size()))

        return context, attn_scores, next_state



