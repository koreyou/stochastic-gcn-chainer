import copy

import cython
import numpy as np
import scipy.sparse as sp
cimport numpy as np
from libcpp.vector cimport vector
from cpython cimport Py_buffer


cdef extern from "<utility>" namespace "std" nogil:
    T move[T](T)


cdef extern from "sampling_c.hpp":
    void c_construct_random_propagation_matrix(
      const float * const in_data, const int * const in_indices,
      const int * const in_indptr, const long unsigned int in_indptr_size,
      const int n_samples, vector[float] &out_data,
      vector[int] &out_indices, vector[int] &out_indptrs)


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
    data, indices, indptr = construct_random_propagation_matrix_impl(
        adj.data, adj.indices, adj.indptr, n_samples
    )
    return sp.csr_matrix((data, indices, indptr), shape=adj.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
def construct_random_propagation_matrix_impl(
        np.ndarray[float, ndim=1, mode="c"] in_data not None,
        np.ndarray[int, ndim=1, mode="c"] in_indices not None,
        np.ndarray[int, ndim=1, mode="c"] in_indptr not None,
        int n_samples):
    cdef long unsigned int in_indptr_size = len(in_indptr)
    cdef vector[float] out_data
    cdef vector[int] out_indices, out_indptrs
    c_construct_random_propagation_matrix(
      &in_data[0], &in_indices[0],
      &in_indptr[0], in_indptr_size,
      n_samples, out_data, out_indices, out_indptrs)

    cdef ArrayWrapperFloat out_data_w = ArrayWrapperFloat()
    out_data_w.set_data(out_data)
    out_data_np = np.asarray(out_data_w)

    cdef ArrayWrapperInt out_indices_w = ArrayWrapperInt()
    out_indices_w.set_data(out_indices)
    out_indices_np = np.asarray(out_indices_w)

    cdef ArrayWrapperInt out_indptrs_w = ArrayWrapperInt()
    out_indptrs_w.set_data(out_indptrs)
    out_indptrs_np = np.asarray(out_indptrs_w)

    return out_data_np, out_indices_np, out_indptrs_np


cdef class ArrayWrapperInt(object):
    cdef vector[int] vec
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    # constructor and destructor are fairly unimportant now since
    # vec will be destroyed automatically.

    cdef set_data(self, vector[int]& data):
       self.vec = move(data)

    # now implement the buffer protocol for the class
    # which makes it generally useful to anything that expects an array
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])

        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(int)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'i'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        """need this even though not used
        """
        pass


cdef class ArrayWrapperFloat(object):
    cdef vector[float] vec
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    # constructor and destructor are fairly unimportant now since
    # vec will be destroyed automatically.

    cdef set_data(self, vector[float]& data):
       self.vec = move(data)

    # now implement the buffer protocol for the class
    # which makes it generally useful to anything that expects an array
    def __getbuffer__(self, Py_buffer *buffer, float flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])

        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(float)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'f'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        """need this even though not used
        """
        pass
