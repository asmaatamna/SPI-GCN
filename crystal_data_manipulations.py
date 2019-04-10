from __future__ import division
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as data
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.io import loadmat
from itertools import chain
import os
import sys
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import networkx as nx
from scipy.stats import multivariate_normal

def pdf(poscar, radius=4.):
    """
    Pair Distribution Funtion (PDF). Given a poscar file,
    computes for each atom the distances of its neighbors within
    a given radius
    """
    res = [] # list over dict because there may be many atoms of the same element in a POSCAR
    struct = mg.Structure.from_file(poscar)
    all_neighbors = struct.get_all_neighbors(radius)
    for site, site_neighbors in zip(struct, all_neighbors):
        neighbors = {}
        site_neighbors.sort(key=lambda x: x[-1])
        for s, d in site_neighbors:
            if s.specie.symbol in neighbors:
                # neighbors[s.specie.symbol].append((s.to_unit_cell, d))
                neighbors[s.specie.symbol].append((s, d))
            else:
                # neighbors[s.specie.symbol] = [(s.to_unit_cell, d)]
                neighbors[s.specie.symbol] = [(s, d)]
            # neighbors.append((s.specie.symbol, d))
        # res.append((site.specie.symbol, neighbors))
        res.append((site, neighbors))
    return res


def first_neighbors(poscar, radius=4.):
    """
    Returns the list of lists of first neighbors of atoms
    in a POSCAR file, poscar. An atom may have many first neighbors
    (i.e. at the same distance).

    TODO: change interface? 'struct' instead of 'poscar' as input?
    """
    res = []
    struct = mg.Structure.from_file(poscar)
    all_neighbors = struct.get_all_neighbors(radius)
    for neighbors in all_neighbors:
        min_d = min([x[1] for x in neighbors])
        min_distances = [i for i, x in enumerate(neighbors) if abs(x[-1] - min_d) < 1e-13]
        res.append([neighbors[i] for i in min_distances])
    return res

def get_periodic_site(site, sites):
    """
    Given a periodic site (of type pymatgen.core.sites.PeriodicSite), returns
    the index of the equivalent periodic site in the list sites.
    """
    for i, s in enumerate(sites):
        if site.is_periodic_image(s):
            return i
    return -1

def get_graph_representation(poscar, atomic_symbols, is_cartesian=True):
    # TODO: Does not seem to work when using both graph and cartesian coordinates.
    # A possible fix would be to only use graph of nearest neighbors + atoms type.
    # For the GAN, in order not to break backprop, we would need to compute the adjacency
    # matrix from the cartesian coordianates with Python (i.e. classic Euclidean norm) instead of using Pymatgen.

    """
    Returns the graph representation corresponding to a crystal (its "elementary" cell), i.e.
    an adjacency matrix A, a vertices matrix X, and a lattice parameters matrix L.
    The vertices correspond to elementary atoms. The edges symbolize connections between
    an atom (vertex) and its first neighbors. Parameter d corresponds to the dimension
    of the vectors encoding atoms (vertices).
    
    Note that we manipulate "Cartesian" POSCAR files
    """
    # Construct adjacency matrix A
    # TODO: Nearest neighbors calculation must be changed
    struct = mg.Structure.from_file(poscar)
    L = struct.lattice.matrix
    nb_vertices = len(struct)
    d = len(atomic_symbols)
    A = np.zeros(shape=(nb_vertices, nb_vertices)) # Adjacency matrix (two atoms are adjacent if they're
    X = np.zeros(shape=(nb_vertices, d)) # Nodes attribes matrix. Each row contains the atom's type
    Coord = np.zeros(shape=(nb_vertices, 3)) # Cartesian coordinates of atoms in the POSCAR
    neighbors = first_neighbors(poscar)
    for i in range(nb_vertices):
        for j in range(len(neighbors[i])):
            idx = get_periodic_site(neighbors[i][j][0], struct)
            if 11 < 3:
                A[i][idx] += 1.
            if 1 < 3: # Here, we will have a "classic" adjacency matrix with all entries either 1 or 0 (instead of a multigraph)
                A[i][idx] = 1.
                A[idx][i] = 1.
    # Construct features matrix X
    for i in range(nb_vertices):
        X[i][atomic_symbols.index(struct[i].species_string)] = 1.
        # X[i][len(atomic_symbols):] = struct[i].coords

    # Costruct the Cartesian coordinates matrix
    # TODO: Not used as of now. If needed, Coord must be added in the return statement
    for i in range(nb_vertices):
        Coord[i] = struct[i].coords
    
    # Not sure this function is needed anywhere, though it might be useful at some point...
    # symbols = struct.symbol_set
    
    # If the POSCAR is in "Cartesian" format, the lattice parameters
    # have already been used to calculate atoms absolute coordinates
    if is_cartesian:
        return A, X
    return A, X, L

def graph_to_poscar(A, X, L, atomic_symbols, filename="POSCAR", save_file=False):
    """
    Transforms a graph to a POSCAR string. The adjacency matrix is not needed here as it is deduced from
    the features matrix X.
    """
    nb_atomic_symbols = len(atomic_symbols)
    components = dict.fromkeys(atomic_symbols) # TODO: Must be a dictionary
    # Extract atom sets per atom type
    # We consider that 
    for j in range(nb_atomic_symbols):
        components[atomic_symbols[j]] = np.array([X[i][nb_atomic_symbols:] for i in range(X.shape[0]) if np.argmax(X[i][:nb_atomic_symbols]) == j])
    # Prepare POSCAR content
    res = ""
    # Prepare symbols line
    s = ""
    for i in range(len(components)):
        if len(components[atomic_symbols[i]]) > 0:
            s += atomic_symbols[i] + " "
    s = s[:-1]
    s += "\n"
    res += s
    res += str(1.0) + "\n"
    
    for a, b, c in L:
        res += "    %.8f    %.8f    %.8f\n" % (a, b, c)
    res += s
    line = ""
    for i in range(len(components)):
        block = components[atomic_symbols[i]]
        if len(block) > 0:
            line += "    " + str(len(block))
            
    line += "\n"
    res += line
    res += "Cartesian\n"
    for i in range(len(components)):
        block = components[atomic_symbols[i]]
        for x, y, z in block:
            res += "    %.8f    %.8f    %.8f\n" % (x, y, z)
            
    # Save POSCAR file
    if save_file:
        poscar = open(filename, "w")
        poscar.write(res)
        poscar.close()
    return res

def corrected_poscar(poscar, new_poscar):
    """
    This function reformats the POSCAR files of our database so that
    'struct = mg.Structure.from_file(poscar)' detects the correct atomic symbols
    in the POSCAR. To do so, we only need to duplicate the first line in the POSCAR (e.g. 'H Ni')
    before the 6th line corresponding to the number of atoms in the elementary cell.
    
    poscar: str. Name of the file we want to reformat.
    new_poscar: str. Name of the new POSCAR file to create.
    
    Note: We assume all POSCAR files have the same structure up to the 6th line.
    """
    my_file = open(poscar, "r")
    content = my_file.readlines()
    new_content = [x for x in content[:5]]
    new_content.append(content[0])
    new_content += content[5:]
    my_file.close()
    my_new_file = open(new_poscar, "w")
    for x in new_content:
        my_new_file.write(x)
    my_new_file.close()

def reduce_graph(A, X):
    """
    This function removes disconnected (isolated) vertices of a graph
    represented by its adjacency matrix A and its features matrix X.
    """
    # Create a networkx graph from adjacency matrix A
    G = nx.Graph(data=A)
    # Remove disconnected vertices
    G.remove_nodes_from(nx.isolates(G))
    return nx.to_numpy_matrix(G), X[G.nodes()]

def minimal_pair_distances(data, is_poscar=False):
    """
    Takes as input a POSCAR file or a graph corresponding to a POSCAR file
    (adj. mat. A and feat. mat. X) and returns the list of minimal
    distances between its atoms.
    """
    res = []
    if not is_poscar:
        A, X = data
        poscar_str = graph_to_poscar(A, X)
        structure = mg.Structure.from_str(poscar_str, "POSCAR")
    else:
        structure = mg.Structure.from_file(data)
    for i in range(len(structure) - 1):
        for j in range(i + 1, len(structure)):
            res.append(structure.get_distance(i, j))
    return res

def corrected_poscar_main(metal, nb_cif_files):
    # Code to reformat POSCAR files in the database
    # Note: Paths must be adapted to needs
    data_directories = ['../Asmas-data-and-scripts/BASE-DFT_cart/' + metal + '/', '../Asmas-data-and-scripts/cif2poscar_data/' + metal + '/']
    # TODO: Number of POSCARs in each directory should be computed automatically
    nb_files = [30, nb_cif_files]
    target_directory = 'Crystallographic_databases/' + metal + '_corrected_POSCARS/'
    idx = 0
    for i, directory in enumerate(data_directories):
        if i == 0:
            filename = "POSCAR.cart"
        else:
            filename = "POSCAR"
        for j in range(nb_files[i]):
            idx += 1
            corrected_poscar(directory + str(j + 1) + "/" + filename, target_directory + "POSCAR_" + str(idx))