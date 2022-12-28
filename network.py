import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.metrics import roc_auc_score


# This tutorial builds a model consisting of two
# `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
# new node representations by averaging neighbor information. DGL provides
# ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.
#
# The model then predicts the probability of existence of an edge by
# computing a score between the representations of both incident nodes
# with a function (e.g. an MLP or a dot product), which you will see in
# the next section.


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv0 = nn.AdaptiveMaxPool1d(in_feats)
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv3 = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')

        # AttributeError: 'GraphSAGE' object has no attribute 'bias'
        # self.initialize_weights()

    def forward(self, g_, in_feat):
        h_ = self.conv0(in_feat)
        h_ = self.conv1(g_, h_)
        h_ = torch.tanh(h_)
        h_ = F.dropout(h_, 0.5)
        h_ = self.conv2(g_, h_)
        h_ = torch.tanh(h_)
        h_ = self.conv3(g_, h_)
        return h_

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Module):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


# The following module produces a scalar score on each edge
# by concatenating the incident nodes’ features and passing it to an MLP.


class MLPPredictor(nn.Module):  # Multi-Layer-Perceptron
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats * 4)
        self.W2 = nn.Linear(h_feats * 4, h_feats * 2)
        self.W3 = nn.Linear(h_feats * 2, 1)

        # self.initialize_weights()

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h_ = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W3(torch.relu(self.W2(torch.relu(self.W1(h_))))).squeeze(1)}

    def forward(self, g_, h_):
        with g_.local_scope():
            g_.ndata['h'] = h_
            g_.apply_edges(self.apply_edges)
            return g_.edata['score']

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


class DotPredictor(nn.Module):
    def forward(self, g_, h_):
        with g_.local_scope():
            g_.ndata['h'] = h_
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            # print('u_dot_v', fn.u_dot_v('h', 'h', 'score'))
            g_.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g_.edata['score'][:, 0]


def compute_loss(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_]).to('cuda')
    labels = torch.cat([torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])]).to('cuda')
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)
