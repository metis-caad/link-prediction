# link-prediction

AI-based autocompletion of graph-based architectural spatial configurations using link prediction with graph neural networks (GNN).

The main goal of the approach is to estimate the probability of relations between the rooms of the spatial configuration graph using the relevant semantic information.

The approach is published in the context of the paper ["Autocompletion of Design Data in Semantic
Building Models using Link Prediction and Graph Neural
Networks"](http://papers.cumincad.org/cgi-bin/works/paper/ecaade2022_222) submitted and presented @ [eCAADe 2022](https://kuleuven.ecaade2022.be/).

# Requirements

Following packages are required to run link prediction:

`python3`

`python3-pip`

[Deep Graph Library (DGL)](https://www.dgl.ai/pages/start.html)

[CUDA](https://developer.nvidia.com/cuda-zone)

[Matplotlib](https://matplotlib.org/stable/users/getting_started/)

# Run evaluation

Run `./train.sh` in console.

You shoud get results similar to the evaluation in the paper (see linked paper).

The evaluation was tested on Ubuntu 22.04 LTS with Python 3.10 and CUDA 11.6.
