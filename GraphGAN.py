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
# import pymatgen as mg
# from pymatgen.io.vasp.inputs import Poscar
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
            nn.ReLU(), # nn.Tanh() # nn.ReLU()
            # nn.Dropout(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        y = self.deterministic_output(x)
        return y

class MLP3(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, out_dim):
        super(MLP3, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.BatchNorm1d(h1_dim, track_running_stats=False),
            nn.Tanh(),  # nn.Tanh(), nn.ReLU()
            nn.Linear(h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim, track_running_stats=False),
            nn.Tanh(),  # nn.Tanh(), nn.ReLU()
            nn.Linear(h2_dim, h3_dim),
            nn.BatchNorm1d(h3_dim, track_running_stats=False),
            nn.Tanh(),  # nn.Tanh(), nn.ReLU()
            nn.Linear(h3_dim, out_dim)
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
        # d: dimension of the feature space, i.e. the size of a vector x representing an atom
        # in the POSCAR
        # f: dimension of the embedding space, i.e. the size of a vector x' representing an atom
        # in the feature space
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
        res = F.softmax(self.W_1(torch.matmul(norm_A, res)), dim=-1) # F.softmax(self.W_1(torch.matmul(norm_A, res)), dim=-1) # F.tanh(self.W_1(torch.matmul(norm_A, res)))
        graph_embedding = res.sum(dim=-2) # res.sum(dim=-2) # res.max(dim=-2)[0] # res.sum(dim=-2) # New addition for classical conv.: keepdim=True
        return graph_embedding # .transpose(-2, -1) # Was: graph_embedding

    def normalize_adjacency_matrix(self, A):
        assert A.shape[0] == A.shape[1], "Error: Adjacency matrix must be a square matrix"
        res = A.clone() + torch.eye(A.shape[0], dtype=torch.float64)
        res = res / res.sum(1, keepdim=True)
        return res

class ConvolutionDiscriminator(nn.Module):
    """
    The discriminator's architecture is inspired from the Graph Convolutional Network
    introduced by Kipf et al., ICLR 2017.
    """

    def __init__(self, d, f):
        # d: dimension of the feature space, i.e. the size of a vector x representing an atom
        # in the POSCAR
        # f: dimension of the embedding space, i.e. the size of a vector x' representing an atom
        # in the feature space
        super(EnergyModule, self).__init__()

        # Graph embedding module
        self.graph_conv = GraphConvolutionModule(d, f)

        # Once the graph embedding is done, we need to predict either
        # it is real (1) or fake (0) using a MLP
        self.mlp = MLP(f, 8 * f, 1)

    def forward(self, A, X):
        graph_embedding = self.graph_conv(A, X)
        res = F.sigmoid(self.mlp(graph_embedding))
        return res

class GenericGraphGenerator(nn.Module):

    def __init__(self, f, nb_vertices, dim):
        super(GenericGraphGenerator, self).__init__()
        self.nb_vertices = nb_vertices
        self.dim = dim

        self.mlp_A = MLP3(f, 128, 256, 512, nb_vertices * (nb_vertices - 1) // 2)
        self.mlp_X = MLP3(f, 128, 256, 512, nb_vertices * dim)

    def forward(self, z):
        A = F.sigmoid(self.mlp_A(z))
        X = self.mlp_X(z)

        if len(A.shape) > 1 and len(X.shape) > 1:
            list_graphs = [self.transform_to_graph(a, x) for a, x in zip(A, X)]
            A_res = torch.stack([x[0] for x in list_graphs])
            X_res = torch.stack([x[1] for x in list_graphs])
        else:
            A_res, X_res = self.transform_to_graph(A, X)

        return A_res, X_res

    def transform_to_graph(self, A, X):
        """
        Format the generated outputs to graph format. In particular, transforms the upper triangular
        probabilistic adjacency matrix to an adjacency matrix, deletes disconnected vertices both from
        adjacency matrix and from features matrix.
        """
        u_tri = torch.zeros(self.nb_vertices, self.nb_vertices, dtype=torch.float64)
        # Transform the output array (A) into a symmetric probabilistic adjacency matrix
        u_tri[np.triu_indices(self.nb_vertices, 1)] = A
        res_A = u_tri + torch.t(u_tri)
        res_X = X.view(-1, self.dim)
        return res_A, res_X

class CrystalGenerator(nn.Module):
    """
    Generator of crystal data.
    """

    def __init__(self, f, nb_atoms, nb_symbols):
        super(CrystalGenerator, self).__init__()
        self.nb_atoms = nb_atoms
        self.nb_symbols = nb_symbols

        self.mlp_X_coord = MLP(f, 8 * f, nb_atoms * 3) # MLP3(f, 128, 256, 512, nb_atoms * 3)
        self.mlp_X_atom = MLP(f, 8 * f, nb_atoms * nb_symbols) # MLP3(f, 128, 256, 512, nb_atoms * nb_symbols)
        self.mlp_L = MLP(f, 8 * f, 3 * 3) # MLP3(f, 128, 256, 512, 3 * 3)  # Lattice parameters: 3 vectors of dimension 3

    def forward(self, z):
        X_coord = self.mlp_X_coord(z).view(-1, 3)
        # TODO: Not sure the reshaping is correct/the best way to do it
        X_atom = F.softmax(self.mlp_X_atom(z).view(-1, self.nb_atoms, self.nb_symbols), dim=-1) # TODO: check wheter dim. is correct
        L = self.mlp_L(z).view(-1, 3)

        if 11 < 3:
            if len(A.shape) > 1 and len(X_coord.shape) > 1 and len(X_atom.shape) > 1 and len(L.shape) > 1:
                list_graphs = [self.transform_to_graph(a, x_coord, x_atom, l) for a, x_coord, x_atom, l in
                               zip(A, X_coord, X_atom, L)]
                A_res = torch.stack([x[0] for x in list_graphs])
                X_res = torch.stack([x[1] for x in list_graphs])
                L_res = torch.stack([x[2] for x in list_graphs])
            else:
                A_res, X_res, L_res = self.transform_to_graph(A, X_coord, X_atom, L)

        return torch.cat((X_atom, X_coord, L), dim=1) # X_coord, X_atom, L

    def transform_to_graph(self, X_coord,  X_atom, L):
        """
        Format the generated outputs to graph format. In particular, transforms the upper triangular
        probabilistic adjacency matrix to an adjacency matrix, deletes disconnected vertices both from
        adjacency matrix and from features matrix.
        """
        u_tri = torch.zeros(self.nb_atoms, self.nb_atoms, dtype=torch.float64)
        # Transform the output array (A) into a symmetric probabilistic adjacency matrix
        u_tri[np.triu_indices(self.nb_atoms, 1)] = A
        res_A = u_tri + torch.t(u_tri)
        cart_X_coord = torch.matmul(X_coord.view(-1, 3), L.view(3, 3))
        return res_A, torch.cat((X_atom.view(self.nb_atoms, -1), cart_X_coord), dim=1), L.view(3, 3)  # torch.cat((X_atom, cart_X_coord), dim=1), L.view(3, 3)

class ClassificationModule(nn.Module):
    """
    Neural network for graph classification (binary and more).
    """

    def __init__(self, d, f, out):
        # d: dimension of the feature space, i.e. the size of a vector x representing an atom
        # in the POSCAR
        # f: dimension of the embedding space, i.e. the size of a vector x' representing an atom
        # in the feature space
        super(ClassificationModule, self).__init__()
        # Make sure out > 0
        assert out > 0, "Error: output size must be larger than zero"
        self.out = out
        # Graph embedding module
        self.graph_conv = GraphConvolutionModule(d, f)
        # Classical convolution layer
        # ksize_conv = 3
        # self.conv1d = nn.Conv1d(1, 1, ksize_conv)
        # Classical max-pooling layer
        # ksize_pool = 2
        # self.max_pool = nn.MaxPool1d(ksize_pool)
        # Simple MLP for prediction
        in_dim = f # int(math.floor(((f - ksize_conv + 1) - ksize_pool) / ksize_pool + 1))
        self.mlp = MLP(in_dim, 8 * in_dim, out)

    def forward(self, A, X):
        res = self.graph_conv(A, X)
        # res = F.relu(self.conv1d(res))
        # res = self.max_pool(res)

        # If binary classification problem:
        # res = F.sigmoid(self.mlp(graph_embedding))
        #
        # If larger than binary:
        # res = F.softmax(self.mlp(graph_embedding), dim=-1)
        if self.out == 1:
            res = F.sigmoid(self.mlp(res))
        else:
            res = F.softmax(self.mlp(res), dim=-1)
        return res

# "Penalty" functions to be added to the error function during training
# in order to reinforce the learning process

def geom_cons_violation(A, X, lb=1.8, ub=3.):
    """
    Given a crystal graph (A, X), transforms it into POSCAR format
    and returns the distance between the vector of minimal pair distances
    of the POSCAR and a lower- or upper-bound

    Parameters:
    - A, X: Adj. mat. and feat. mat. of a crystal graph
    - lb, ub: lower and upper bound respectively
    """
    distances = cdm.minimal_pair_distances((A, X))
    return np.linalg.norm(distances - lb * np.ones(len(distances))), np.linalg.norm(
        distances - ub * np.ones(len(distances)))