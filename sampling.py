import copy

import numpy as np
import scipy.sparse as sp


def random_sampling(adj, rf, n_layers, n_samples):
    """ Construct the receptive fields and random propagation matrices.
    FIXME: This impl. is not efficient. Re-implement it with C.

    Args:
        adj (scipy.sparse.csr_matrix): Base adjacency matrix of shape (N, N)
        mask (numpy.ndarray): (N, ) length vector that represents the
            receptive field on the last layer
        n_layers (int): Number of GCN layers
        n_samples (int): Number of neighbors to sample (apart from the target
            node itself)

    Returns:
        list of scipy.sparse.csr_matrix: `n_layers`-length random adjacency
            matrices. Staring from the bottom layers.
        list of scipy.sparse.csr_matrix: `n_layers`-length adjacency matrices
            that contains all neighbors.
        list of numpy.ndarray: `n_layers`-length receptive fields for
            sampled nodes, each with shape (N, ).
        list of numpy.ndarray: `n_layers`-length receptive fields for
            all nodes.
    """
    rf_sample_l = rf.astype(bool)

    adjs_sample = []
    adjs = []
    rfs_sample = []
    rfs = []

    adj_diag = adj.diagonal()
    adj_n = copy.deepcopy(adj)
    adj_n.setdiag(0.)
    adj_n.eliminate_zeros()
    for _ in range(n_layers):
        adj_sample_l = construct_random_propagation_matrix(adj_n, n_samples)
        adj_sample_l.setdiag(adj_diag)

        rf_sample_ln1 = adj_sample_l.T.dot(rf_sample_l).astype(bool)
        rf_ln1 = adj.T.dot(rf_sample_l).astype(bool)

        # Slicing for some reason convert matrix to np.float64
        adj_sample_l = adj[rf_sample_l, :][:, rf_sample_ln1].astype(np.float32)
        adj_l = adj[rf_sample_l, :][:, rf_ln1].astype(np.float32)

        adjs_sample.append(adj_sample_l)
        adjs.append(adj_l)
        rfs_sample.append(rf_sample_ln1)
        rfs.append(rf_ln1)
        rf_sample_l = rf_sample_ln1
        del adj_sample_l, adj_l

    return adjs_sample[::-1], adjs[::-1], rfs_sample[::-1], rfs[::-1]


def construct_random_propagation_matrix(adj, n_samples):
    indices = []
    data = []
    # Do no use arange for nodes that has less neighbors than n_samples
    indptrs = [0]
    for j in range(len(adj.indptr) - 1):
        cols = adj.indices[adj.indptr[j]:adj.indptr[j + 1]]
        idx = np.sort(np.random.permutation(len(cols))[:n_samples])
        indices.append(cols[idx])
        # #neighbors / #samples according to the original paper.
        n_D = len(cols) / float(len(idx))
        data.append(adj.data[adj.indptr[j]:adj.indptr[j + 1]][idx] * n_D)
        indptrs.append(indptrs[-1] + len(idx))
    indices = np.concatenate(indices)
    data = np.concatenate(data)
    indptrs = np.array(indptrs)
    return sp.csr_matrix((data, indices, indptrs), shape=adj.shape)
