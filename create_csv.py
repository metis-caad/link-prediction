import hashlib
import json
import os
import sys
import xml.etree.ElementTree as eT
import config


def get_room_type(n_id, _graph):
    for node_el in _graph.findall(namespace + 'node'):
        if n_id == node_el.get('id'):
            data_el = node_el.findall(namespace + 'data')
            for d in data_el:
                if d.get('key') == 'roomType':
                    return d.text.upper()


def get_room_code_number(n_id, _graph):
    room_type = get_room_type(n_id, _graph)
    return config.room_types[config.room_type_codes[str(room_type).upper()]]


namespace = '{http://graphml.graphdrawing.org/xmlns}'
basedir = os.path.dirname(os.path.realpath(sys.argv[0]))
# dirname = os.path.dirname(os.path.realpath(sys.argv[0])) + '/dataset_small/'
datadir = basedir + '/dataset_clean/'
classes = os.listdir(datadir)

nodes = []
room_types = []
rooms = [['id', 'type', 'class']]
edges = [['source', 'target', 'weight']]
feats = []

count_r = 0
count_e = 0

err_file = 0
err_node_s = 0
err_node_t = 0

for cls in classes:
    agraphmls = os.listdir(datadir + '/' + cls + '/')
    for agraphml in agraphmls:
        full_filename = datadir + '/' + cls + '/' + agraphml
        tree = eT.parse(full_filename)
        root = tree.getroot()
        graph = root[0]
        corridor_found = False
        corridor_id = ''
        for node in graph.findall(namespace + 'node'):
            for data in node.findall(namespace + 'data'):
                if data.get('key') == 'roomType':
                    node_type = data.text
                    if node_type.upper() == config.CORRIDOR:
                        corridor_found = True
                        corridor_id = node.get('id')
                        break
            if corridor_found:
                break
        if corridor_found:
            try:
                with open(basedir + '/paths/' + agraphml + '-' + corridor_id + '.json', 'r') as json_file:
                    jsn1 = json_file.readlines()[0]
                lengths = json.loads(jsn1)
            except FileNotFoundError:
                err_file += 1
                continue
            try:
                with open(basedir + '/connectivity/' + cls + '/' + agraphml + '.json', 'r') as json_file:
                    jsn2 = json_file.readlines()[0]
                conns = json.loads(jsn2)
            except FileNotFoundError:
                err_file += 1
                continue
            room_ids = [r.get('id') for r in graph.findall(namespace + 'node')]
            edge_ids = [e.get('id') for e in graph.findall(namespace + 'edge')]
            count_r += len(room_ids)
            count_e += len(edge_ids)
            for edge in graph.findall(namespace + 'edge'):
                source_id = edge.get('source')
                target_id = edge.get('target')
                source_length = 0
                target_length = 0
                try:
                    source_length = lengths[source_id]
                except KeyError:
                    err_node_s += 1
                    pass
                try:
                    target_length = lengths[target_id]
                except KeyError:
                    err_node_t += 1
                    pass
                source_conn = 1
                target_conn = 1
                try:
                    source_conn = conns[source_id]
                except KeyError:
                    err_node_s += 1
                    pass
                try:
                    target_conn = conns[target_id]
                except KeyError:
                    err_node_t += 1
                    pass
                source_type = get_room_type(source_id, graph)
                target_type = get_room_type(target_id, graph)
                if source_type not in room_types:
                    room_types.append(source_type)
                if target_type not in room_types:
                    room_types.append(target_type)
                source = agraphml + source_id + source_type
                target = agraphml + target_id + target_type
                source_code_id = int(hashlib.md5(source.encode('utf-8')).hexdigest(), 16)
                target_code_id = int(hashlib.md5(target.encode('utf-8')).hexdigest(), 16)
                if source_code_id not in nodes:
                    nodes.append(source_code_id)
                if target_code_id not in nodes:
                    nodes.append(target_code_id)
                source_index = str(nodes.index(source_code_id))
                target_index = str(nodes.index(target_code_id))
                source_feat = cls + '@' + str(source_length) + '@' + str(source_conn)
                target_feat = cls + '@' + str(target_length) + '@' + str(target_conn)
                # source_feat = str(source_length) + str(source_conn)
                # target_feat = str(target_length) + str(target_conn)
                if source_feat not in feats:
                    feats.append(source_feat)
                if target_feat not in feats:
                    feats.append(target_feat)
                source_entry = [source_index, source_type, source_feat]
                target_entry = [target_index, target_type, target_feat]
                if source_entry not in rooms:
                    rooms.append(source_entry)
                if target_entry not in rooms:
                    rooms.append(target_entry)
                edge_type = ''
                for data in edge.findall(namespace + 'data'):
                    if data.get('key') == 'edgeType':
                        edge_type = data.text
                edge_entry = [source_index, target_index, str(config.edge_type_weights[edge_type.upper()])]
                edges.append(edge_entry)

# assert count_r == (len(rooms) - 1) and count_e == (len(edges) - 1)  # -1: without header
print(str(len(rooms) - 1) + ' rooms,', str(len(edges) - 1) + ' edges')
print(room_types)
print('ERR lengths file:', err_file, ', ERR node s:', err_node_s, ', ERR node t:', err_node_t)

with open('rooms.csv', 'a+') as rooms_csv:
    for room in rooms:
        rooms_csv.write(','.join(room) + '\n')

with open('edges.csv', 'a+') as edges_csv:
    for edge in edges:
        edges_csv.write(','.join(edge) + '\n')

with open('feat_count.txt', 'w') as fc_txt:
    fc_txt.write(str(len(feats)))
