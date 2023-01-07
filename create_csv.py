import hashlib
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as eT
from argparse import ArgumentParser

import config

parser = ArgumentParser()
parser.add_argument('-t', '--type', dest='csv_type', required=True)
args = parser.parse_args()

csv_type = args.csv_type


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
if csv_type == 'request':
    basedir = basedir + '/requests'
datadir = basedir + '/dataset_clean/'
classes = os.listdir(datadir)

nodes = []
rooms = [['id', 'type', 'class']]
edges = [['source', 'target', 'weight']]
feats = []
feats_counts = {}

nodes_eval = []
eval_rooms = [['id', 'type', 'class']]
eval_edges = [['source', 'target', 'weight']]
eval_feats = []
eval_agraphmls = {}

with_isolated = []
with open(basedir + '/with_isolated.list', 'r') as with_isolated_txt:
    for line in with_isolated_txt.readlines():
        with_isolated.append(line.replace('\n', ''))


def make_csv_entries(write=True):
    count_r = 0
    count_e = 0

    err_file = 0
    err_node_s = 0
    err_node_t = 0

    count_eval = 0
    count_train = 0
    for cls in classes:
        agraphmls = os.listdir(datadir + '/' + cls + '/')
        if not write:
            random.shuffle(agraphmls)
        for agraphml in agraphmls:
            if ('dataset_clean/' + cls + '/' + agraphml) not in with_isolated:
                full_filename = datadir + cls + '/' + agraphml
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
                    if write:
                        # add the agraphml either to training or evaluation
                        for_eval = False
                        if agraphml in eval_agraphmls.values():
                            # add to evaluation
                            eval_filename = full_filename.replace('dataset_clean', 'evaluation')
                            shutil.copy(full_filename, eval_filename)
                            for_eval = True
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
                            source = agraphml + source_id + source_type
                            target = agraphml + target_id + target_type
                            source_code_id = int(hashlib.md5(source.encode('utf-8')).hexdigest(), 16)
                            target_code_id = int(hashlib.md5(target.encode('utf-8')).hexdigest(), 16)
                            if not for_eval:
                                if source_code_id not in nodes:
                                    nodes.append(source_code_id)
                                if target_code_id not in nodes:
                                    nodes.append(target_code_id)
                            else:
                                if source_code_id not in nodes_eval:
                                    nodes_eval.append(source_code_id)
                                if target_code_id not in nodes_eval:
                                    nodes_eval.append(target_code_id)
                            if not for_eval:
                                source_index = str(nodes.index(source_code_id))
                                target_index = str(nodes.index(target_code_id))
                            else:
                                source_index = str(nodes_eval.index(source_code_id))
                                target_index = str(nodes_eval.index(target_code_id))
                            source_feat = cls + '@' + str(source_length) + '@' + str(source_conn)
                            target_feat = cls + '@' + str(target_length) + '@' + str(target_conn)
                            if not for_eval:
                                if source_feat not in feats:
                                    feats.append(source_feat)
                                    feats_counts[source_feat] = 1
                                else:
                                    cnt_s = feats_counts[source_feat]
                                    feats_counts[source_feat] = cnt_s + 1
                                if target_feat not in feats:
                                    feats.append(target_feat)
                                    feats_counts[target_feat] = 1
                                else:
                                    cnt_t = feats_counts[target_feat]
                                    feats_counts[target_feat] = cnt_t + 1
                            else:
                                if source_feat not in eval_feats:
                                    eval_feats.append(source_feat)
                                if target_feat not in eval_feats:
                                    eval_feats.append(target_feat)
                            source_entry = [source_index, source_type, source_feat]
                            target_entry = [target_index, target_type, target_feat]
                            if not for_eval:
                                if source_entry not in rooms:
                                    rooms.append(source_entry)
                                if target_entry not in rooms:
                                    rooms.append(target_entry)
                            else:
                                if source_entry not in eval_rooms:
                                    eval_rooms.append(source_entry)
                                if target_entry not in eval_rooms:
                                    eval_rooms.append(target_entry)
                            edge_type = ''
                            for data in edge.findall(namespace + 'data'):
                                if data.get('key') == 'edgeType':
                                    edge_type = data.text
                            edge_entry = [source_index, target_index, str(config.edge_type_weights[edge_type.upper()])]
                            if not for_eval:
                                edges.append(edge_entry)
                            else:
                                eval_edges.append(edge_entry)
                        if not for_eval:
                            count_train += 1
                        else:
                            count_eval += 1
                    else:
                        for edge in graph.findall(namespace + 'edge'):
                            source_id = edge.get('source')
                            target_id = edge.get('target')
                            source_length = lengths[source_id]
                            target_length = lengths[target_id]
                            source_conn = conns[source_id]
                            target_conn = conns[target_id]
                            source_feat = cls + '@' + str(source_length) + '@' + str(source_conn)
                            target_feat = cls + '@' + str(target_length) + '@' + str(target_conn)
                            if source_feat not in eval_agraphmls.keys():
                                eval_agraphmls[source_feat] = agraphml
                            if target_feat not in eval_agraphmls.keys():
                                eval_agraphmls[target_feat] = agraphml
            else:
                if write:
                    print('Found isolated', agraphml)
    if write:
        print('Train: ' + str(count_train) + ', Eval: ' + str(count_eval))
        print('Train: ' + str(len(rooms) - 1) + ' rooms,', str(len(edges) - 1) + ' edges')
        print('Eval: ' + str(len(eval_rooms) - 1) + ' rooms,', str(len(eval_edges) - 1) + ' edges')
        for feat, count in feats_counts.items():
            if count == 1:
                print('Found 1x available feature', feat)
        print('ERR lengths file:', err_file, ', ERR node s:', err_node_s, ', ERR node t:', err_node_t)


if csv_type == 'eval':
    make_csv_entries(write=False)

make_csv_entries()

if csv_type == 'eval':
    assert len(eval_agraphmls) == len(feats)

with open(basedir + '/rooms.csv', 'a+') as rooms_csv:
    for room in rooms:
        rooms_csv.write(','.join(room) + '\n')

with open(basedir + '/edges.csv', 'a+') as edges_csv:
    for edge_ in edges:
        edges_csv.write(','.join(edge_) + '\n')

with open(basedir + '/feat_count.txt', 'w') as fc_txt:
    fc_txt.write(str(len(feats)))

if csv_type == 'eval':
    with open(basedir + '/queries/rooms.csv', 'a+') as rooms_csv_eval:
        for er in eval_rooms:
            rooms_csv_eval.write(','.join(er) + '\n')

    with open(basedir + '/queries/edges.csv', 'a+') as edges_csv_eval:
        for ee in eval_edges:
            edges_csv_eval.write(','.join(ee) + '\n')

    with open(basedir + '/feat_count_eval.txt', 'w') as fc_eval_txt:
        fc_eval_txt.write(str(len(eval_agraphmls)))
