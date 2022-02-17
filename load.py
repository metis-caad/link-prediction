import dgl
import pandas as pd
import torch

from dataset_cls import QueryDataset
from network import GraphSAGE, MLPPredictor

loaded_graph, labels_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
g = loaded_graph[0]
g = g.to('cuda')
# g_e =

# rooms_csv_pd = pd.read_csv('./queries/test1/rooms.csv')
# rooms_csv_pd.head()
# edges_csv_pd = pd.read_csv('./queries/test1/edges.csv')
# edges_csv_pd.head()
# dataset = QueryDataset('./queries/test1/rooms.csv', './queries/test1/edges.csv')
# graph = dataset[0]
# graph = graph.to('cuda')
# query_graph = dgl.remove_edges(graph, [0])

# Model and predictor
# model = GraphSAGE(query_graph.ndata['feat'].shape[1], 16).to('cuda')
# model.load_state_dict(torch.load('model.pth'))
# predictor = MLPPredictor(16).to('cuda')
#
# outputs = model(query_graph, g.ndata['feat'])
# outputs = model(query_graph, query_graph.ndata['feat'])
# pos_score = predictor(graph, outputs)

