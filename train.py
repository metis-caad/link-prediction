import itertools

import torch

from room_conf_graph import RoomConfGraph
from network import GraphSAGE, MLPPredictor, compute_loss, compute_auc, DotPredictor

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

room_conf_graph = RoomConfGraph()
room_conf_graph.init_graph()

# Model and predictor
model = GraphSAGE(room_conf_graph.train_g.ndata['feat'].shape[1], 16).to('cuda')
# predictor = MLPPredictor(16).to('cuda')
predictor = DotPredictor().to('cuda')

# Optimizer
optimizer = torch.optim.RAdam(itertools.chain(model.parameters(), predictor.parameters()), lr=0.01)
outputs = None
for e in range(500):
    # forward
    outputs = model(room_conf_graph.train_g, room_conf_graph.train_g.ndata['feat'])
    pos_score = predictor(room_conf_graph.train_pos_g, outputs)
    neg_score = predictor(room_conf_graph.train_neg_g, outputs)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 10 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

assert outputs is not None

with torch.no_grad():
    pos_score = predictor(room_conf_graph.test_pos_g, outputs)
    neg_score = predictor(room_conf_graph.test_neg_g, outputs)
    print('AUC Test', compute_auc(pos_score, neg_score))

PATH = 'model.pth'
torch.save(model.state_dict(), PATH)
