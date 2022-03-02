import hashlib
import json
import os
import random
import shutil
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

count_eval = 0
count_train = 0

eval_rooms = [['id', 'type', 'class']]
eval_edges = [['source', 'target', 'weight']]


def make_eval_entries(_room_types_eval, _cls):
    max_conn = len(_room_types_eval)
    max_depth = 3
    evr = []
    for i in range(0, len(_room_types_eval)):
        rte = _room_types_eval[i]
        init_depth = 1
        if rte == config.CORRIDOR:
            init_depth = 0
        for j in range(init_depth, max_depth + 1):
            for k in range(1, max_conn):
                cr_eval = len(eval_rooms) - 1
                eval_rooms.append([str(cr_eval), rte.upper(), cls + '@' + str(j) + '@' + str(k)])
                evr.append([str(cr_eval), rte.upper() + '_' + str(i), cls + '@' + str(j) + '@' + str(k)])
    for entry_o in evr:
        t1 = entry_o[1]
        for entry_i in evr:
            t2 = entry_i[1]
            if t1 != t2:
                eval_edges.append([entry_o[0], entry_i[0], '0.5'])
                eval_edges.append([entry_o[0], entry_i[0], '1'])


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
            # add the agraphml either to training or evaluation
            if agraphmls.index(agraphml) == 0:
                # add to evaluation
                eval_filename = full_filename.replace('dataset_clean', 'evaluation')
                shutil.copy(full_filename, eval_filename)
                count_eval += 1
                room_types_eval = []
                for node_eval in graph.findall(namespace + 'node'):
                    room_type_eval = get_room_type(node_eval.get('id'), graph)
                    room_types_eval.append(room_type_eval)
                make_eval_entries(room_types_eval, cls)
                continue
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
            count_train += 1

# assert count_r == (len(rooms) - 1) and count_e == (len(edges) - 1)  # -1: without header
print('Train: ' + str(count_train) + ', Eval: ' + str(count_eval))
print('Train: ' + str(len(rooms) - 1) + ' rooms,', str(len(edges) - 1) + ' edges')
print('ERR lengths file:', err_file, ', ERR node s:', err_node_s, ', ERR node t:', err_node_t)

with open('rooms.csv', 'a+') as rooms_csv:
    for room in rooms:
        rooms_csv.write(','.join(room) + '\n')

with open('edges.csv', 'a+') as edges_csv:
    for edge in edges:
        edges_csv.write(','.join(edge) + '\n')

with open('feat_count.txt', 'w') as fc_txt:
    fc_txt.write(str(len(feats)))

with open('queries/rooms.csv', 'a+') as rooms_csv_eval:
    for er in eval_rooms:
        rooms_csv_eval.write(','.join(er) + '\n')

with open('queries/edges.csv', 'a+') as edges_csv_eval:
    for ee in eval_edges:
        edges_csv_eval.write(','.join(ee) + '\n')
