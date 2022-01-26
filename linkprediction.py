import dgl.data
import itertools
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.metrics import roc_auc_score

######################################################################
# Overview of Link Prediction with GNN
# ------------------------------------
#
# Many applications such as social recommendation, item recommendation,
# knowledge graph completion, etc., can be formulated as link prediction,
# which predicts whether an edge exists between two particular nodes. This
# tutorial shows an example of predicting whether a citation relationship,
# either citing or being cited, between two papers exists in a citation
# network.
#
# This tutorial formulates the link prediction problem as a binary classification
# problem as follows:
#
# -  Treat the edges in the graph as *positive examples*.
# -  Sample a number of non-existent edges (i.e. node pairs with no edges
#    between them) as *negative* examples.
# -  Divide the positive examples and negative examples into a training
#    set and a test set.
# -  Evaluate the model with any binary classification metric such as Area
#    Under Curve (AUC).
#
# .. note::
#
#    The practice comes from
#    `SEAL <https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf>`__,
#    although the model here does not use their idea of node labeling.
#
# In some domains such as large-scale recommender systems or information
# retrieval, you may favor metrics that emphasize good performance of
# top-K predictions. In these cases you may want to consider other metrics
# such as mean average precision, and use other negative sampling methods,
# which are beyond the scope of this tutorial.

loaded_graph, labels_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
g = loaded_graph[0]
g = g.to('cuda')

# This tutorial randomly picks 10% of the edges for positive examples in
# the test set, and leave the rest for the training set. It then samples
# the same number of edges for negative examples in both sets.

# Split edge set for training and testing
u, v = g.edges()
# u = all FROMs tensor([    0,     1,     2,  ..., 16633, 16634, 16635])
# v = all TOs tensor([    1,     2,     3,  ..., 16626, 16626, 16632])

eids = np.arange(g.number_of_edges())  # [    0     1     2 ... 21313 21314 21315]
eids = np.random.permutation(eids)  # [17621 14611  7158 ...  5487  8340 17927]

test_size = int(len(eids) * 0.1)  # 2131
train_size = g.number_of_edges() - test_size  # 19185
assert train_size + test_size == g.number_of_edges()

test_pos_u = u[eids[:test_size]]  # TEST FROMs tensor([7308, 9614, 9578,  ..., 7720, 2053, 8687])
test_pos_v = v[eids[:test_size]]  # TEST TOs tensor([7310, 9612, 9575,  ..., 7718, 2051, 8689])

train_pos_u = u[eids[test_size:]]  # TRAIN FROMs tensor([14658, 13537,  2549,  ...,  7673,  6313,  5475])
train_pos_v = v[eids[test_size:]]  # TRAIN TOs tensor([14657, 13540,  2548,  ...,  7674,  6311,  5474])

# Find all negative edges and split them for training and testing
# adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())),  # (21316, (all FROMs, all TOs))
                    shape=(g.number_of_nodes(), g.number_of_nodes()))
# (0, 1)        1.0 = exists
# (1, 2)        1.0
# (2, 3)        1.0
# :     :
# (16633, 16626)  1.0
# (16634, 16626)  1.0
# (16635, 16632)  1.0

# 1 here is g.number_of_nodes() X g.number_of_nodes() matrix of 1.0s
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
# [[0. 0. 1. ... 1. 1. 1.]
#  [1. 0. 0. ... 1. 1. 1.]
#  [1. 1. 0. ... 1. 1. 1.]
#  ...
#  [1. 1. 1. ... 0. 1. 1.]
#  [1. 1. 1. ... 1. 0. 1.]
#  [1. 1. 1. ... 1. 1. 0.]]
neg_u, neg_v = np.where(adj_neg != 0)
# neg_u array([    0     0     0 ... 16635 16635 16635])
# neg_v array([    2     3     6 ... 16631 16633 16634])
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())  # (276718544, 21316)
# [ 78648025  31491672 159151284 ... 269168643 163136799 105657104]
test_neg_u = neg_u[neg_eids[:test_size]]  # TEST NEGATIVE FROMs [12094   102  8713 ...  2998 12125  4583]
test_neg_v = neg_v[neg_eids[:test_size]]  # TEST NEGATIVE TOs [16284 16232  6220 ... 10584 14511  1145]

train_neg_u = neg_u[neg_eids[test_size:]]  # TRAIN NEGATIVE FROMs [15299  9499  2560 ... 13913 15639 13647]
train_neg_v = neg_v[neg_eids[test_size:]]  # TRAIN NEGATIVE TOs [13526 12575 10998 ...  2479  6414  1084]

# When training, you will need to remove the edges in the test set from
# the original graph. You can do this via ``dgl.remove_edges``.
#
# .. note::
#
#    ``dgl.remove_edges`` works by creating a subgraph from the
#    original graph, resulting in a copy and therefore could be slow for
#    large graphs. If so, you could save the training and test graph to
#    disk, as you would do for preprocessing.
#

train_g = dgl.remove_edges(g, eids[:test_size])

# This tutorial builds a model consisting of two
# `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
# new node representations by averaging neighbor information. DGL provides
# ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')

    def forward(self, g_, in_feat):
        h_ = self.conv1(g_, in_feat)
        h_ = F.relu(h_)
        h_ = self.conv2(g_, h_)
        return h_


# The model then predicts the probability of existence of an edge by
# computing a score between the representations of both incident nodes
# with a function (e.g. an MLP or a dot product), which you will see in
# the next section.


# Positive graph, negative graph, and ``apply_edges``
# ---------------------------------------------------
#
# In previous tutorials you have learned how to compute node
# representations with a GNN. However, link prediction requires you to
# compute representation of *pairs of nodes*.
#
# DGL recommends you to treat the pairs of nodes as another graph, since
# you can describe a pair of nodes with an edge. In link prediction, you
# will have a *positive graph* consisting of all the positive examples as
# edges, and a *negative graph* consisting of all the negative examples.
# The *positive graph* and the *negative graph* will contain the same set
# of nodes as the original graph.  This makes it easier to pass node
# features among multiple graphs for computation.  As you will see later,
# you can directly feed the node representations computed on the entire
# graph to the positive and the negative graphs for computing pair-wise
# scores.
#
# The following code constructs the positive graph and the negative graph
# for the training set and the test set respectively.

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes()).to('cuda')
# Graph(num_nodes=16636, num_edges=19185,
#       ndata_schemes={}
#       edata_schemes={})

train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes()).to('cuda')
# Graph(num_nodes=16636, num_edges=19185,
#       ndata_schemes={}
#       edata_schemes={})

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes()).to('cuda')
# Graph(num_nodes=16636, num_edges=2131,
#       ndata_schemes={}
#       edata_schemes={})
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes()).to('cuda')
# Graph(num_nodes=16636, num_edges=2131,
#       ndata_schemes={}
#       edata_schemes={})

# The benefit of treating the pairs of nodes as a graph is that you can
# use the ``DGLGraph.apply_edges`` method, which conveniently computes new
# edge features based on the incident nodes’ features and the original
# edge features (if applicable).
#
# DGL provides a set of optimized builtin functions to compute new
# edge features based on the original node/edge features. For example,
# ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’
# representations for each edge.


# The following module produces a scalar score on each edge
# by concatenating the incident nodes’ features and passing it to an MLP.
#

class MLPPredictor(nn.Module):  # Multi-Layer-Perceptron
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

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
        return {'score': self.W2(F.relu(self.W1(h_))).squeeze(1)}

    def forward(self, g_, h_):
        with g_.local_scope():
            g_.ndata['h'] = h_
            g_.apply_edges(self.apply_edges)
            return g_.edata['score']


# Model and predictor
model = GraphSAGE(train_g.ndata['feat'].shape[1], 16).to('cuda')
predictor = MLPPredictor(16).to('cuda')


def compute_loss(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_]).to('cuda')
    labels = torch.cat([torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])]).to('cuda')
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)


# Optimizer
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=0.01)
trained = None
for e in range(3000):
    # forward
    trained = model(train_g, train_g.ndata['feat'])
    pos_score = predictor(train_pos_g, trained)
    neg_score = predictor(train_neg_g, trained)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

assert trained is not None

with torch.no_grad():
    pos_score = predictor(test_pos_g, trained)
    neg_score = predictor(test_neg_g, trained)
    print('AUC', compute_auc(pos_score, neg_score))

