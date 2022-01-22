import hashlib
import os
import sys
import xml.etree.ElementTree as eT


def get_room_type(room_id, _graph):
    for room in _graph.findall(namespace + 'node'):
        if room_id == room.get('id'):
            data = room.findall(namespace + 'data')
            for d in data:
                if d.get('key') == 'roomType':
                    return d.text


namespace = '{http://graphml.graphdrawing.org/xmlns}'
dirname = os.path.dirname(os.path.realpath(sys.argv[0])) + '/dataset/'
classes = os.listdir(dirname)

rooms = [['id', 'class', 'type']]
edges = [['source', 'target', 'weight']]

count_r = 0
count_e = 0

for cls in classes:
    agraphmls = os.listdir(dirname + '/' + cls + '/')
    for agraphml in agraphmls:
        full_filename = dirname + '/' + cls + '/' + agraphml
        tree = eT.parse(full_filename)
        root = tree.getroot()
        graph = root[0]
        room_ids = [r.get('id') for r in graph.findall(namespace + 'node')]
        edge_ids = [e.get('id') for e in graph.findall(namespace + 'edge')]
        count_r += len(room_ids)
        count_e += len(edge_ids)
        for edge in graph.findall(namespace + 'edge'):
            source_id = edge.get('source')
            target_id = edge.get('target')
            source_type = get_room_type(source_id, graph)
            target_type = get_room_type(target_id, graph)
            source = agraphml + source_id + source_type
            target = agraphml + target_id + target_type
            source_code_id = str(int(hashlib.md5(source.encode('utf-8')).hexdigest(), 16))
            target_code_id = str(int(hashlib.md5(target.encode('utf-8')).hexdigest(), 16))
            # print(source_code, target_code)
            source_entry = [source_code_id, cls, source_type]
            target_entry = [target_code_id, cls, target_type]
            if source_entry not in rooms:
                rooms.append(source_entry)
            if target_entry not in rooms:
                rooms.append(target_entry)
            edge_entry = [source_code_id, target_code_id, '1']
            edges.append(edge_entry)

assert count_r == (len(rooms) - 1) and count_e == (len(edges) - 1)  # -1: without header
print(str(len(rooms) - 1) + ' rooms,', str(len(edges) - 1) + ' edges')

with open('rooms.csv', 'a+') as rooms_csv:
    for room in rooms:
        rooms_csv.write(','.join(room) + '\n')

with open('edges.csv', 'a+') as edges_csv:
    for edge in edges:
        edges_csv.write(','.join(edge) + '\n')
