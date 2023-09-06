import dgl
import torch

from dataset_cls import QueryDataset
from network import GraphSAGE, MLPPredictor, DotPredictor, compute_auc
from room_conf_graph import RoomConfGraph

device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_graph, labels_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
g = loaded_graph[0]
g = g.to(device)

dataset_eval = QueryDataset('./queries/rooms.csv', './queries/edges.csv')
room_conf_graph_eval = dataset_eval[0]
print('Eval graph', room_conf_graph_eval)
dgl.save_graphs('room_conf_graph_eval.dgl', room_conf_graph_eval)

room_conf_graph_eval = RoomConfGraph('_eval')
room_conf_graph_eval.init_graph()

# Model and predictor
model = GraphSAGE(g.ndata['feat'].shape[1], 16).to(device)
model.load_state_dict(torch.load('model.pth'))
# predictor = MLPPredictor(16).to(device)
predictor = DotPredictor().to(device)

outputs = model(room_conf_graph_eval.train_g, room_conf_graph_eval.train_g.ndata['feat'])
with torch.no_grad():
    pos_score = predictor(room_conf_graph_eval.train_pos_g, outputs)
    neg_score = predictor(room_conf_graph_eval.train_neg_g, outputs)
    print('AUC Eval', compute_auc(pos_score, neg_score))

# print(pos_score)
# outputs = model(query_graph, query_graph.ndata['feat'])
# pos_score = predictor(query_graph, outputs)
