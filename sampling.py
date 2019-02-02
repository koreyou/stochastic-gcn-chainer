import copy

import numpy as np
import scipy.sparse as sp


def random_sampling(adj, mask, n_layers, n_samples):
    mask = mask.astype(bool)
    adjs = []
    adj_diag = adj.diagonal()
    adj_n = copy.deepcopy(adj)
    adj_n.setdiag(0.)
    adj_n.eliminate_zeros()
    for _ in range(n_layers):
        indices = []
        data = []
        # Do no use arange for nodes that has less neighbors than n_samples
        indptrs = [0]
        for j in range(len(adj_n.indptr) - 1):
            cols = adj_n.indices[adj_n.indptr[j]:adj_n.indptr[j + 1]]
            idx = np.sort(np.random.permutation(len(cols))[:n_samples])
            indices.append(cols[idx])
            data.append(adj_n.data[adj_n.indptr[j]:adj_n.indptr[j + 1]][idx])
            indptrs.append(indptrs[-1] + len(idx))
        indices = np.concatenate(indices)
        data = np.concatenate(data)
        indptrs = np.array(indptrs)
        adj_i = sp.csr_matrix((data, indices, indptrs), shape=adj.shape)
        adj_i.setdiag(adj_diag)
        mask_new = adj_i.T.dot(mask).astype(bool)
        mask_others = adj_n.T.dot(mask).astype(bool)
        # Do NOT use adj after pruning. Instead we only apply pruning to find
        # sampled nodes, and use all connections that use the sampled nodes.
        mask_others[mask_new] = False
        adj_i_others = adj[mask, :][:, mask_others]
        adj_i = adj[mask, :][:, mask_new]
        adjs.append((adj_i, adj_i_others, mask, mask_others))
        mask = mask_new
        del adj_i_others, adj_i
    return adjs[::-1], mask
