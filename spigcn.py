# GraphGAN architecture

from __future__ import division
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as data
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.io import loadmat
from itertools import chain
import os
import sys
import networkx as nx
from scipy.stats import multivariate_normal
import crystal_data_manipulations as cdm

# Personalized Multilayer Perceptrons
# TODO: Should be better to pass the desired number of
# hidden layers as a parameter of a unique MLP module
class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim, track_running_stats=False),
            nn.Tanh()
            # nn.Dropout(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        y = self.deterministic_output(x)
        return y

# Module implementing graph convolutions
class GraphConvolutionModule(nn.Module):
    """
    Neural network to predict energetic property (enthalpy) of a (binary, for the moment) metal hydride
    represented as a graph (adjacency matrix + features matrix).

    This architecture is inspired from Kipf et al., ICLR 2017 (graph embedding).
    """

    def __init__(self, d, f):
        # d: dimension of the feature space, i.e. the size of a vector x representing a vertex
        # f: dimension of thevertex embedding space
        super(GraphConvolutionModule, self).__init__()
        h =  128
        self.W_0 = nn.Linear(d, h, bias=False)
        self.W_1 = nn.Linear(h, f, bias=False)

    def forward(self, A, X):
        if 1 < 3:
            if len(A.shape) > 2:
                norm_A = torch.stack([self.normalize_adjacency_matrix(a) for a in A])
            else:
                norm_A = self.normalize_adjacency_matrix(A)
        else:
            norm_A = A
        res = F.tanh(self.W_0(torch.matmul(norm_A, X)))
        res = F.softmax(self.W_1(torch.matmul(norm_A, res)), dim=-1)
        graph_embedding = res.sum(dim=-2) 
        return graph_embedding

    def normalize_adjacency_matrix(self, A):
        assert A.shape[0] == A.shape[1], "Error: Adjacency matrix must be a square matrix"
        res = A.clone() + torch.eye(A.shape[0], dtype=torch.float64)
        res = res / res.sum(1, keepdim=True)
        return res

class ClassificationModule(nn.Module):
    """
    Neural network for graph classification (binary and more).
    """

    def __init__(self, d, f, out):
        # d: dimension of the feature space
        # f: dimension of the embedding space
        # out: number of classes
        super(ClassificationModule, self).__init__()
        # Make sure out > 0
        assert out > 0, "Error: output size must be larger than zero"
        self.out = out
        # Graph embedding module
        self.graph_conv = GraphConvolutionModule(d, f)
        self.mlp = MLP(f, 8 * f, out)

    def forward(self, A, X):
        res = self.graph_conv(A, X)

        # If binary classification problem:
        # res = F.sigmoid(self.mlp(res))
        #
        # If more than two classes:
        # res = F.softmax(self.mlp(res), dim=-1)
        if self.out == 1:
            res = F.sigmoid(self.mlp(res))
        else:
            res = F.softmax(self.mlp(res), dim=-1)
        return res
