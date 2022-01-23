import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset

rooms_csv = pd.read_csv('./rooms.csv')
rooms_csv.head()

edges_csv = pd.read_csv('./edges.csv')
edges_csv.head()


class RoomConfDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='room_conf_data')

    def process(self):
        nodes_data = pd.read_csv('./rooms.csv')
        edges_data = pd.read_csv('./edges.csv')
        node_features = torch.from_numpy(nodes_data['type'].astype('category').cat.codes.to_numpy())
        node_labels = torch.from_numpy(nodes_data['class'].astype('category').cat.codes.to_numpy())
        edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_sources = torch.from_numpy(edges_data['source'].to_numpy())
        edges_targets = torch.from_numpy(edges_data['target'].to_numpy())

        self.graph = dgl.graph((edges_sources, edges_targets), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


dataset = RoomConfDataset()
room_conf_graph = dataset[0]

print(room_conf_graph)

dgl.save_graphs('room_conf_graph.dgl', room_conf_graph)

loaded_graph, label_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
print(loaded_graph)
print(label_dict)
