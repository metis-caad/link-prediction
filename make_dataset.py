import dgl
import pandas as pd

from dataset_cls import RoomConfDataset

rooms_csv = pd.read_csv('./rooms.csv')
rooms_csv.head()

edges_csv = pd.read_csv('./edges.csv')
edges_csv.head()

dataset = RoomConfDataset()
room_conf_graph = dataset[0]

print(room_conf_graph)

dgl.save_graphs('room_conf_graph.dgl', room_conf_graph)

loaded_graph, label_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
print(loaded_graph)
print(label_dict)
