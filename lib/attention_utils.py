import numpy as np
import networkx as nx
from scipy import sparse
from IPython.core.debugger import Tracer

def feature_calculator(window_size, graph):
    """
    Calculating the feature tensor.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :return target_matrices: Target tensor.
    """
    keys = graph.nodes()
    new_values = [i for i in np.arange(len(keys))]
    dictionary = dict(zip(keys, new_values))
    index_1 = [dictionary[edge[0]] for edge in graph.edges()]
    index_2 = [dictionary[edge[1]] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = len(graph.nodes())
    adjacency_matrix = sparse.coo_matrix((values, (index_1, index_2)),
                                         shape=(node_count, node_count),
                                         dtype=np.float32)

    #Tracer()()
    degrees = adjacency_matrix.sum(axis=0)[0].tolist()
    degs = sparse.diags(degrees, [0])
    normalized_adjacency_matrix = np.exp(-1*(degs-adjacency_matrix).todense())
    target_matrices = [normalized_adjacency_matrix]
    powered_A = normalized_adjacency_matrix
    if window_size > 1:
        for power in (range(window_size-1)):
            powered_A = np.exp(-power*(degs-adjacency_matrix).todense())
            to_add = powered_A
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices

def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetworkX object.
    :return adjacency_matrix_opposite: Indicator matrix.
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite