"""
Link Prediction using Graph Neural Networks
===========================================

In the :doc:`introduction <1_introduction>`, you have already learned
the basic workflow of using GNNs for node classification,
i.e. predicting the category of a node in a graph. This tutorial will
teach you how to train a GNN for link prediction, i.e. predicting the
existence of an edge between two arbitrary nodes in a graph.

By the end of this tutorial you will be able to

-  Build a GNN-based link prediction model.
-  Train and evaluate the model on a small DGL-provided dataset.

(Time estimate: 28 minutes)

"""

# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp

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
# -  Sample a number of non-existent edges (i.e. node pairs with no edges
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
#
# Loading graph and features
# --------------------------
#
# Following the :doc:`introduction <1_introduction>`, this tutorial
# first loads the Cora dataset.
#

import dgl.data

loaded_graph, labels_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
g = loaded_graph[0]

######################################################################
# Prepare training and testing sets
# ---------------------------------
#
# This tutorial randomly picks 10% of the edges for positive examples in
# the test set, and leave the rest for the training set. It then samples
# the same number of edges for negative examples in both sets.
#

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
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),  # (21316, (all FROMs tensor, all TOs tensor))
                    shape=(g.number_of_nodes(), g.number_of_nodes()))
# (0, 1)        1.0 = exists
# (1, 2)        1.0
# (2, 3)        1.0
# :     :
# (16633, 16626)  1.0
# (16634, 16626)  1.0
# (16635, 16632)  1.0
#
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

######################################################################
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

######################################################################
# Define a GraphSAGE model
# ------------------------
#
# This tutorial builds a model consisting of two
# `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
# new node representations by averaging neighbor information. DGL provides
# ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.
#

from dgl.nn.pytorch.conv import SAGEConv


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        # print(in_feats, type(in_feats))
        # print(h_feats, type(h_feats))
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')

    def forward(self, g_, in_feat):
        h_ = self.conv1(g_, in_feat)
        h_ = F.relu(h_)
        h_ = self.conv2(g_, h_)
        return h_


######################################################################
# The model then predicts the probability of existence of an edge by
# computing a score between the representations of both incident nodes
# with a function (e.g. an MLP or a dot product), which you will see in
# the next section.
#
# .. math::
#
#
#    \hat{y}_{u\sim v} = f(h_u, h_v)
#


######################################################################
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
#

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
# Graph(num_nodes=16636, num_edges=19185,
#       ndata_schemes={}
#       edata_schemes={})

train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
# Graph(num_nodes=16636, num_edges=19185,
#       ndata_schemes={}
#       edata_schemes={})

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
# Graph(num_nodes=16636, num_edges=2131,
#       ndata_schemes={}
#       edata_schemes={})
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
# Graph(num_nodes=16636, num_edges=2131,
#       ndata_schemes={}
#       edata_schemes={})

######################################################################
# The benefit of treating the pairs of nodes as a graph is that you can
# use the ``DGLGraph.apply_edges`` method, which conveniently computes new
# edge features based on the incident nodes’ features and the original
# edge features (if applicable).
#
# DGL provides a set of optimized builtin functions to compute new
# edge features based on the original node/edge features. For example,
# ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’
# representations for each edge.
#

import dgl.function as fn


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


######################################################################
# You can also write your own function if it is complex.
# For instance, the following module produces a scalar score on each edge
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


######################################################################
# .. note::
#
#    The builtin functions are optimized for both speed and memory.
#    We recommend using builtin functions whenever possible.
#
# .. note::
#
#    If you have read the :doc:`message passing
#    tutorial <3_message_passing>`, you will notice that the
#    argument ``apply_edges`` takes has exactly the same form as a message
#    function in ``update_all``.
#


######################################################################
# Training loop
# -------------
#
# After you defined the node representation computation and the edge score
# computation, you can go ahead and define the overall model, loss
# function, and evaluation metric.
#
# The loss function is simply binary cross entropy loss.
#
# The evaluation metric in this tutorial is AUC.
#

# print(train_g.ndata['feat'].shape)
# exit()

model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(16)
# pred = DotPredictor()


def compute_loss(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_])
    labels = torch.cat([torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score_, neg_score_):
    scores = torch.cat([pos_score_, neg_score_]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score_.shape[0]), torch.zeros(neg_score_.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


######################################################################
# The training loop goes as follows:
#
# .. note::
#
#    This tutorial does not include evaluation on a validation
#    set. In practice you should save and evaluate the best model based on
#    performance on the validation set.
#

# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(3000):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))

# Thumbnail credits: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
