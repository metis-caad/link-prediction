import dgl
import torch

from dataset_cls import RequestDataset
from network import GraphSAGE, MLPPredictor, DotPredictor, compute_auc
from room_conf_graph import RoomConfGraph

loaded_graph, labels_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
g = loaded_graph[0]
g = g.to('cuda')

dataset_req = RequestDataset('./requests/rooms.csv', './requests/edges.csv')
room_conf_graph_req = dataset_req[0].to('cuda')
print('Request graph', room_conf_graph_req)
dgl.save_graphs('room_conf_graph_req.dgl', room_conf_graph_req)

room_conf_graph_req = RoomConfGraph('_req')
room_conf_graph_req.init_graph()

# Model and predictor
model = GraphSAGE(g.ndata['feat'].shape[1], 16).to('cuda')
model.load_state_dict(torch.load('model.pth'))
# predictor = MLPPredictor(16).to('cuda')
predictor = DotPredictor().to('cuda')

outputs = model(room_conf_graph_req.train_g, room_conf_graph_req.train_g.ndata['feat'])
print(outputs)

with torch.no_grad():
    pos_score = predictor(room_conf_graph_req.train_pos_g, outputs)
    neg_score = predictor(room_conf_graph_req.train_neg_g, outputs)
    print('AUC Request', compute_auc(pos_score, neg_score))
