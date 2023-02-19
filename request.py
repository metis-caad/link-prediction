import os
import sys

import dgl
import torch

from dataset_cls import RequestDataset
from network import GraphSAGE, MLPPredictor, DotPredictor, compute_auc
from room_conf_graph import RoomConfGraph

basedir = os.path.dirname(os.path.realpath(sys.argv[0]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_graph, labels_dict = dgl.load_graphs(basedir + '/room_conf_graph.dgl', [0])
g = loaded_graph[0]
g = g.to(device)

dataset_req = RequestDataset(basedir + '/requests/rooms.csv', basedir + '/requests/edges.csv')
room_conf_graph_req = dataset_req[0].to(device)
# print('Request graph', room_conf_graph_req)
dgl.save_graphs(basedir + '/room_conf_graph_req.dgl', room_conf_graph_req)

room_conf_graph_req = RoomConfGraph('_req')
room_conf_graph_req.init_graph()

# Model and predictor
model = GraphSAGE(g.ndata['feat'].shape[1], 16).to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(basedir + '/model.pth'))
else:
    # Load model on CPU, then move to GPU, dgl creates fc_self.bias on GPU by default, hacky solution to handle this
    x = torch.load(basedir + '/model.pth', map_location=torch.device('cpu'))
    x['conv1.fc_self.bias'] = x['conv1.bias']
    x['conv2.fc_self.bias'] = x['conv2.bias']
    x['conv3.fc_self.bias'] = x['conv3.bias']
    del x['conv1.bias']
    del x['conv2.bias']
    del x['conv3.bias']
    model.load_state_dict(x)
# predictor = MLPPredictor(16).to('cuda')
predictor = DotPredictor().to(device)

outputs = model(room_conf_graph_req.train_g, room_conf_graph_req.train_g.ndata['feat'])
# print(outputs)

with torch.no_grad():
    pos_score = predictor(room_conf_graph_req.train_pos_g, outputs)
    neg_score = predictor(room_conf_graph_req.train_neg_g, outputs)
    # print('AUC Request:')
    print(compute_auc(pos_score, neg_score))
