import dgl

from dataset_cls import RoomConfDataset

dataset = RoomConfDataset()
room_conf_graph = dataset[0]

print(room_conf_graph)

dgl.save_graphs('room_conf_graph.dgl', room_conf_graph)

loaded_graph, label_dict = dgl.load_graphs('room_conf_graph.dgl', [0])
print(loaded_graph)
print(label_dict)
