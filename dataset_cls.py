import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset


def get_feat_count(feat_type=''):
    with open('feat_count' + feat_type + '.txt', 'r') as fc_txt:
        feat_count = int(fc_txt.readlines()[0])
    assert feat_count > 0
    return feat_count


class RoomConfDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='room_conf_data')

    def process(self):
        nodes_data = pd.read_csv('./rooms.csv')
        edges_data = pd.read_csv('./edges.csv')
        node_features = torch.from_numpy(pd.get_dummies(nodes_data['class'], prefix='type').to_numpy())
        node_labels = torch.from_numpy(pd.get_dummies(nodes_data['type'], prefix='type').to_numpy())
        edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_sources = torch.from_numpy(edges_data['source'].to_numpy())
        edges_targets = torch.from_numpy(edges_data['target'].to_numpy())

        self.graph = dgl.graph((edges_sources, edges_targets), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features.float().resize(nodes_data.shape[0], get_feat_count())
        # TODO is it reasonable to float (sage doesnt work otherwise)?
        self.graph.ndata['label'] = node_labels.long()
        self.graph.edata['weight'] = edge_features

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class QueryDataset(DGLDataset):
    __slots__ = ['rooms_csv', 'edges_csv']

    def __init__(self, rooms_csv_, edges_csv_):
        self.rooms_csv = rooms_csv_
        self.edges_csv = edges_csv_

        super().__init__(name='room_conf_query')

    def process(self):
        nodes_data = pd.read_csv(self.rooms_csv)
        node_features = torch.from_numpy(pd.get_dummies(nodes_data['class'], prefix='type').to_numpy())
        node_labels = torch.from_numpy(pd.get_dummies(nodes_data['type'], prefix='type').to_numpy())
        edges_data = pd.read_csv(self.edges_csv)
        edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_sources = torch.from_numpy(edges_data['source'].to_numpy())
        edges_targets = torch.from_numpy(edges_data['target'].to_numpy())

        self.graph = dgl.graph((edges_sources, edges_targets), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features.float().resize(nodes_data.shape[0], get_feat_count(feat_type='_eval'))
        # TODO is it reasonable to float (sage doesnt work otherwise)?
        self.graph.ndata['label'] = node_labels.long()
        self.graph.edata['weight'] = edge_features

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
