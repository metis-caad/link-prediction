import os
import sys

basedir = os.path.dirname(os.path.realpath(sys.argv[0]))
datadir = basedir + '/evaluation/'
classes = os.listdir(datadir)

with open(basedir + '/eval_clustering_autocompletion.list', 'a+') as eval_list_file:
    for cls in classes:
        agraphmls = os.listdir(datadir + '/' + cls + '/')
        for agraphml in agraphmls:
            eval_filename = 'dataset_clean/' + cls + '/' + agraphml
            eval_list_file.write(eval_filename + '\n')
