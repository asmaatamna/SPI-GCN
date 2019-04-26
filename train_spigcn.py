from __future__ import division
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import sklearn.metrics
from statsmodels.distributions.empirical_distribution import ECDF
import spigcn as spi
import datasets as ds
import crystal_data_manipulations as cdm
import energetic_data_calculations as edc

# Get command line args
#
# Select dataset type to upload:
# 'crystal' for crystallographic data
# 'public' for public graph datasets available on TU Dortmundt website
ds_type = sys.argv[1]
# Dataset name
# Public datasets used are: 
# 	AIDS
# 	BZR
# 	ENZYMES
# 	MUTAG
# 	PTC_MR
# 	IMDB-BINARY
# 	IMDB-MULTI
# 	PROTEINS (TODO)
# 	COLLAB (TODO)
# 	NCI1
#
# For the chemical dataset use:
# 	HYDRIDES
ds_name = sys.argv[2]
runs = int(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5]) # Learning rate for Adam
wd = float(sys.argv[6]) # Weight decay for Adam

if ds_type == 'hydrides':
    # Load crystal training data, transform them to graphs, and store them in a list
    # Also, get the max. graph size (max. number of vertices)
    crystal_graphs = []
    energetic_data = np.empty((0, 4))
    atomic_symbols = ['Ag', 'Ca', 'Cd', 'Co', 'Cu', 'Fe', 'Hf', 'K', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Ni', 'Pd', 'Pt', 'Rh', 'Ru', 'Sc', 'Sr', 'Ta', 'Tc', 'Ti', 'V', 'Y', 'Zn', 'Zr']
    energetic_data_all = edc.get_energetic_data()
    nb_vertices_max = 0 # Maximal number of vertices (atoms) contained in a graph (i.e. POSCAR file)
    inf_idx, sup_idx = 0, len(atomic_symbols)
    for metal in atomic_symbols[inf_idx:sup_idx]:
        poscars_directory = "Crystallographic_databases/" + metal + "_corrected_POSCARS/"
        file_names = os.listdir(poscars_directory)
        for i in range(len(file_names)):
            A, X = cdm.get_graph_representation(poscars_directory + "POSCAR_" + str(i + 1), ['H'] + atomic_symbols[inf_idx:sup_idx], is_cartesian=True)
            if len(A) > nb_vertices_max:
                nb_vertices_max = len(A)
            crystal_graphs.append((A, X))

        # Prepare training data
        energetic_data = np.concatenate((energetic_data, energetic_data_all[metal]), axis=0)
    energetic_data = np.array(energetic_data)
    dataset = ds.EnergyDataset(crystal_graphs, energetic_data[:, 1], energetic_data[:, 3])
    
elif ds_type == 'public':
    dataset = ds.UDortmundGraphDataset(ds_name)

# Determines whether to perform cross validation
cv = True
# Dimension of a node attribute vector
d = dataset[0][1].shape[-1]
# Dimension of embedding space
f = 32

# Define output size for classification (i.e. binary or more)
max_label = int(max([dataset[i][-1] for i in range(len(dataset))]))
out = max_label + 1 if max_label > 1 else 1
test_accuracies = []

# Disable annoying PyTorch warnings
warnings.filterwarnings('ignore')

# Train the neural network
# Performs k-fold cross validation

for run in range(runs):
    if cv:
        # Split dataset into k folds for k-fold cross validation
        k = 10 # Number of data folds
        f_sizes = (k - 1) * [int(k/100. * len(dataset))]
        f_sizes.append(len(dataset) - sum(f_sizes))
        # Set seed
        torch.manual_seed(run)
        folds = random_split(dataset, f_sizes)
    else:
        # If not cv, devide dataset into training and test sets at random
        nb_train_data = int(0.9 * len(dataset))
        train_idx = np.random.choice(len(dataset), size=nb_train_data, replace=False)
        test_idx = np.array([i for i in range(len(dataset)) if i not in train_idx])
    
    # Training
    ntimes = k if cv else 1
    for i in range(ntimes):
        net = spi.ClassificationModule(d, f, out)
        net.double()
        criterion = F.cross_entropy if out > 1 else F.binary_cross_entropy
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        if cv:
            test_dataloader = torch.utils.data.DataLoader(folds[i], batch_size=len(folds[i]))
            # Merge training folds into a unique dataset
            train_folds = torch.utils.data.ConcatDataset([folds[j] for j in range(k) if j != i])
            train_dataloader = torch.utils.data.DataLoader(train_folds, batch_size=len(train_folds))
        else:
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=nb_train_data, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset) - nb_train_data, sampler=torch.utils.data.SubsetRandomSampler(test_idx))

        if ds_type == 'public':
            As, Xs, labels = next(iter(train_dataloader)) # Note: Adapt the number of returned elements to your dataset
            test_As, test_Xs, test_labels = next(iter(test_dataloader))
        if ds_type == 'hydrides':
            As, Xs, _, labels = next(iter(train_dataloader))
            test_As, test_Xs, _, test_labels = next(iter(test_dataloader))
        for epoch in range(epochs):
            net.zero_grad()
            outputs = net(As, Xs)
            loss = criterion(outputs, labels.long()) if out > 1 else criterion(outputs, labels.view(-1, 1)) 
            loss.backward()
            optimizer.step()
            # print('Epoch {}: training error: {:.3f}'.format(epoch, loss))

        net.eval()
        test_outputs = net(test_As, test_Xs)
        if out > 1:
            test_accuracies.append(sklearn.metrics.accuracy_score(test_labels.numpy(), np.argmax(test_outputs.detach().numpy(), axis=1)))
        else:
            test_accuracies.append(sklearn.metrics.accuracy_score(test_labels.numpy(), torch.round(test_outputs).detach().numpy()))

        print('Run {}, fold {}: training error: {:.3f}, test accuracy: {:.3f}'.format(run, i, loss, test_accuracies[-1]))

# Save test accuracies in a .txt file
np.savetxt('./Test-accuracies/' + ds_name + '/' + ds_name + '_test_accuracies_10-fold-cv_' + str(runs) + 'x_' + str(epochs) + '_epochs_lr_' + str(lr) + '_weight_decay_' + str(wd) + '_seed_' + str(run) + '.txt', test_accuracies)
