import torch

from network import GraphSAGE, MLPPredictor

with open('feat_count.txt', 'r') as fc_txt:
    feat_count = int(fc_txt.readlines()[0])
assert feat_count > 0

# Model and predictor
model = GraphSAGE(feat_count, 16).to('cuda')
model.load_state_dict(torch.load('model.pth'))

predictor = MLPPredictor(16).to('cuda')
