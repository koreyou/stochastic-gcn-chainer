import chainer
import chainer.functions as F
import numpy as np
from chainer import initializers
from chainer import reporter
try:
    from cupyx.scipy.sparse import issparse
except ImportError:
    from scipy.sparse import issparse
import scipy.sparse as sp


class GCN(chainer.Chain):
    def __init__(self, adj, features, labels, feat_size, dropout=0.5):
        super(GCN, self).__init__()
        if not sp.isspmatrix_csr(adj):
            raise TypeError(
                'adj must be csr_matrix but %s was given' % str(type(adj)))
        if not sp.isspmatrix_csr(features):
            raise TypeError(
                'features must be csr_matrix but %s was given' % str(type(adj)))

        n_class = np.max(labels) + 1
        with self.init_scope():
            self.gconv1 = GraphConvolution(
                features.shape[1], feat_size, dropout=dropout)
            self.gconv2 = GraphConvolution(
                feat_size, n_class, dropout=dropout)
        self.adj = adj
        self.features = features
        self.labels = labels

    def _forward(self):
        h = F.relu(self.gconv1(self.features, self.adj))
        out = self.gconv2(h, self.adj)
        return out

    def __call__(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        return loss

    def evaluate(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        return float(loss.data), float(accuracy.data)

    def predict(self, idx):
        out = self._forward()
        out = out[idx]
        pred = self.xp.argmax(out.data)
        return pred

    def predict_proba(self, idx):
        out = self._forward()
        out = out[idx]
        return out.data

    def to_gpu(self, device=None):
        import cupyx.scipy.sparse.csr_matrix as xcsr_matrix
        self.adj = xcsr_matrix(self.adj)
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(GCN, self).to_gpu(device=device)

    def to_cpu(self):
        self.adj = self.adj.get()
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(GCN, self).to_cpu()


class GraphConvolution(chainer.Link):
    def __init__(self, in_size, out_size=None, nobias=True, initialW=None,
                 initial_bias=None, dropout=0.5):
        super(GraphConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size
        self.dropout = dropout
        with self.init_scope():
            if initialW is None:
                initialW = initializers.GlorotUniform()
            self.W = chainer.Parameter(initialW, (in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, out_size)

    def __call__(self, x, adj):
        h = sparse_matmul2(adj, x)

        if issparse(h):
            if chainer.config.train:
                with chainer.no_backprop_mode():
                    h.data = F.dropout(h.data, self.dropout).data
                h.eliminate_zeros()
            z = sparse_matmul2(h, self.W)
        else:
            h = F.dropout(h, self.dropout)
            z = F.matmul(h, self.W)

        if self.b is not None:
            z += self.b

        return z


class SparseMatmul2(chainer.Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, inputs):
        if self.a is None and self.b is None:
            a, b = inputs
            self.retain_inputs((0, 1))
            z = a.dot(b)
        elif self.a is None:
            a, = inputs
            z = self.b.T.dot(a.T).T
            self.retain_inputs(tuple())
        elif self.b is None:
            b, = inputs
            a = self.a
            z = a.dot(b)
            self.retain_inputs(tuple())
        else:
            a = self.a
            b = self.b
            z = a.dot(b)
            self.retain_inputs(tuple())
        return z,

    def backward(self, inputs, grad_outputs):
        gz, = grad_outputs
        if self.a is None and self.b is None:
            a, b = inputs
            ga = gz.dot(b.T)
            gb = a.T.dot(gz)
            return ga, gb
        elif self.a is None:
            ga = self.b.dot(gz.T).T
            return ga,
        elif self.b is None:
            gb = self.a.T.dot(gz)
            return gb,


def sparse_matmul2(a, b):
    """ Matmul on two sparse or non-sparse matrices.
    This function calcualtes gradient only on the non-sparse matrices.
    """
    if issparse(a) and issparse(b):
        return a.dot(b)
    elif issparse(a):
        return SparseMatmul2(a, None)(b)
    elif issparse(b):
        return SparseMatmul2(None, b)(a)
    else:
        return SparseMatmul2(None, None)(a, b)