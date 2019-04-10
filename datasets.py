from __future__ import division
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import graph_dataset_dortmund_university as udortmund

# Define customized dataset class
class CrystalGraphDataset(Dataset):
    """
    Note: We don't return the lattice parameters here
    as we don't use them when the POSCAR files are "cartesian".
    """

    def __init__(self, crystal_graphs):
        self.size = len(crystal_graphs)
        self.crystal_graphs = crystal_graphs

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        A, X = self.crystal_graphs[index]
        # Padding with zeros so that all the items of the data set have the same size
        nb_vertices_max = max([len(a[0]) for a in self.crystal_graphs])
        res_A = np.zeros((nb_vertices_max, nb_vertices_max))
        res_X = np.zeros((nb_vertices_max, X.shape[1]))
        res_A[:A.shape[0], :A.shape[1]] = A
        res_X[:X.shape[0], :] = X
        return res_A, res_X

    def __len__(self):
        return self.size

# Define customized synthetic dataset class
class GraphDataset(Dataset):
    """
    Generates a set of 'size' adjacency, features matrices, and lattice parameters
    from a probabilistic adjacency matrix.
    """

    def __init__(self, prob_A, dim, size=1000):
        assert prob_A.shape[0] == prob_A.shape[1], "Error: Adjacency matrix must be a square matrix"
        self.size = size
        self.adjacency_matrices = []
        for i in range(size):
            # We want to generate a symmetric matrix, therefore we only
            # take the upper triangular part of the matrix sampled from prob_A
            # and mirror it
            res = np.triu(np.random.binomial(1, p=prob_A).astype(np.float64), k=1)
            self.adjacency_matrices.append(res + res.transpose())
        # self.adjacency_matrices = [np.random.binomial(1, p=prob_A) for i in range(nb_graphs)]
        self.features_matrices = [np.random.randn(prob_A.shape[0], dim) for i in range(size)]
        self.lattice_parameters = [np.random.randn(3, 3) for i in range(size)]

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.adjacency_matrices[index], self.features_matrices[index], self.lattice_parameters[index]

    def __len__(self):
        return self.size

# Define customized dataset class
class EnergyDataset(Dataset):
    """
    Returns the list of crystals in graph form along with enthalpy values
    and stability labels (1 or 0) for each compound.
    """

    def __init__(self, crystal_graphs, enthalpies, labels):
        assert len(crystal_graphs) == len(enthalpies) and len(crystal_graphs) == len(labels), "Error: input sets should have the same length"
        self.nb_vertices_max = max([len(a[0]) for a in crystal_graphs])
        self.crystal_graphs = np.array(crystal_graphs)
        self.enthalpies = np.array(enthalpies)
        self.labels = np.array(labels)
        self.size = len(self.crystal_graphs)

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        A, X = self.crystal_graphs[index]
        # Padding with zeros so that all the items of the dataset have the same size
        res_A = np.zeros((self.nb_vertices_max, self.nb_vertices_max))
        res_X = np.zeros((self.nb_vertices_max, X.shape[1]))
        res_A[:A.shape[0], :A.shape[1]] = A
        res_X[:X.shape[0], :] = X
        return res_A, res_X, self.enthalpies[index], self.labels[index]

    def __len__(self):
        return self.size

# Define customized dataset class
class UDortmundGraphDataset(Dataset):
    def __init__(self, ds_name):
        # TODO: Implement random selection of training/test data

        As, Xs, labels = udortmund.get_graph_data(ds_name)

        self.nb_vertices_max = max([len(a) for a in As]) # 100
        self.d_max = max([x.shape[1] for x in Xs]) # Maximum features number for a node (different from graph to graph only when node features aren't available)

        self.As = np.array(As)
        self.Xs = np.array(Xs)
        self.labels = np.array(labels)

        self.size = len(self.As)

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        A = self.As[index]
        X = self.Xs[index]
        label = self.labels[index]
        # Normalize adjacency matrix by adding self-loops then dividing by degree matrix
        # A = A + np.identity(A.shape[0])
        # A = A / np.sum(A, axis=1, keepdims=True)
        # Padding with zeros so that all the items of the data set have the same size
        res_A = np.zeros((self.nb_vertices_max, self.nb_vertices_max))
        res_X = np.zeros((self.nb_vertices_max, self.d_max))
        res_A[:A.shape[0], :A.shape[1]] = A[:self.nb_vertices_max, :self.nb_vertices_max]
        res_X[:X.shape[0], :X.shape[1]] = X[:self.nb_vertices_max, :]
        return res_A, res_X, label

    def __len__(self):
        return self.size

# Define "fake" dataset class
class FakeDataset(Dataset):
    """
    Generates a set of fake data consisting of Gaussian vectors of dimension f.
    """

    def __init__(self, size=1000, f=32):
        self.fake_data = np.random.randn(size, f)

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.fake_data[index]

    def __len__(self):
        return len(self.fake_data)
