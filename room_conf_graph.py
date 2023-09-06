import os
import sys

import dgl.data
import numpy as np
import scipy.sparse as sp
import torch


class RoomConfGraph:
    basedir = os.path.dirname(os.path.realpath(sys.argv[0]))

    def __init__(self, feat_type=''):
        self.train_g = None
        self.train_neg_g = None
        self.train_pos_g = None
        self.test_neg_g = None
        self.test_pos_g = None
        self.feat_type = feat_type
        self.device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_graph(self):
        loaded_graph, labels_dict = dgl.load_graphs(self.basedir + '/room_conf_graph' + self.feat_type + '.dgl', [0])
        gg = loaded_graph[0]
        g = dgl.add_reverse_edges(gg)
        g = g.to(self.device)

        # This tutorial randomly picks 10% of the edges for positive examples in
        # the test set, and leave the rest for the training set. It then samples
        # the same number of edges for negative examples in both sets.

        # Split edge set for training and testing
        u, v = g.edges()
        # u = all FROMs tensor([    0,     1,     2,  ..., 16633, 16634, 16635])
        # v = all TOs tensor([    1,     2,     3,  ..., 16626, 16626, 16632])

        eids = np.arange(g.number_of_edges())  # [    0     1     2 ... 21313 21314 21315]
        eids = np.random.permutation(eids)  # [17621 14611  7158 ...  5487  8340 17927]

        test_factor = 0.1
        if self.feat_type != '':
            test_factor = 0
        test_size = int(len(eids) * test_factor)  # 2131
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

        self.train_g = dgl.remove_edges(g, eids[:test_size])

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

        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes()).to(self.device)
        # Graph(num_nodes=16636, num_edges=19185,
        #       ndata_schemes={}
        #       edata_schemes={})

        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes()).to(self.device)
        # Graph(num_nodes=16636, num_edges=19185,
        #       ndata_schemes={}
        #       edata_schemes={})

        if self.feat_type == '':
            self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes()).to(self.device)
            # Graph(num_nodes=16636, num_edges=2131,
            #       ndata_schemes={}
            #       edata_schemes={})
            self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes()).to(self.device)
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

        assert self.train_g is not None
        assert self.train_pos_g is not None
        assert self.train_neg_g is not None
        if self.feat_type == '':
            assert self.test_pos_g is not None
            assert self.test_neg_g is not None
