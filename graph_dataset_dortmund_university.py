import numpy as np
import networkx as nx
import os.path as path

def get_graph_data(ds_name):
    """
    This function processes graph data files downloaded from
    'https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets'
    and transforms them to our desired graph format (adjacency + feature (node attributes) matrices).
    
    Returns:
    - List of adjacency matrices
    - List of node attribute matrices
    - List of graph labels
    
    Example:
        As, Xs, labels = get_graph_data()
    """
    adjacency_matrices = []
    features_matrices = []
    data_dir = './' # Datasets location. To bo changed if needed
    edges = [(x[0], x[1]) for x in np.loadtxt(data_dir + ds_name + '/' + ds_name +  '_A.txt', delimiter=',')]
    nodes = np.loadtxt(data_dir + ds_name + '/' + ds_name +  '_graph_indicator.txt', dtype='int')
    
    # Computing adjacency matrices
    G = nx.Graph()
    G.add_nodes_from(range(1, len(nodes) + 1))
    G.add_edges_from(edges)
    graphs_sizes = []
    for i in range(1, nodes[-1] + 1):
        graphs_sizes.append(len([x for x in nodes if x == i]))
    inf_idx = 1
    for i in range(len(graphs_sizes)):
        G_i = G.subgraph(range(inf_idx, inf_idx + graphs_sizes[i]))
        adjacency_matrices.append(np.array(nx.to_numpy_matrix(G_i)))
        inf_idx += graphs_sizes[i]
    
    # Getting feature (node attribute) matrices if they exist
    if path.isfile('./' + data_dir + ds_name + '/' + ds_name + '_node_attributes.txt'):
        features = np.loadtxt(data_dir + ds_name + '/' + ds_name + '_node_attributes.txt', delimiter=',', ndmin=2)
        inf_idx = 0
        for i in range(len(graphs_sizes)):
            features_matrices.append(features[inf_idx:inf_idx + graphs_sizes[i]])
            inf_idx += graphs_sizes[i]
    else:
        for i in range(len(graphs_sizes)):
            # features_matrices.append(np.ones((graphs_sizes[i], 1)))
            features_matrices.append(np.identity(graphs_sizes[i]))
    
    # Getting graph labels    
    labels = np.loadtxt(data_dir + ds_name + '/' + ds_name +  '_graph_labels.txt')

    # Transform negative labels (-1) to 0 for binary classification problems
    if max(labels) > 1:
        for i in range(len(labels)):
            labels[i] -= 1
    else:
        for i in range(len(labels)):
            if labels[i] < 0:
                labels[i] = 0
    
    return adjacency_matrices, features_matrices, labels

