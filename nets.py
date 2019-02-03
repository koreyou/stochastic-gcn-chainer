import chainer
import chainer.functions as F
import numpy as np
from chainer import initializers
from chainer import reporter
from chainer.datasets import TupleDataset
try:
    from cupyx.scipy.sparse import issparse
except ImportError:
    from scipy.sparse import issparse
import scipy.sparse as sp
from tqdm import tqdm


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
            self.W1 = chainer.Parameter(
                initializers.GlorotUniform(),(features.shape[1], feat_size))
            self.W2 = chainer.Parameter(
                initializers.GlorotUniform(),(feat_size, n_class))
        self.dropout = dropout
        self.adj = adj
        self.features = features
        self.labels = labels
        self.n_samples = 2
        # FIXME: initialize history with proper values
        history = np.random.random(
            (adj.shape[0], feat_size)).astype(np.float32)
        self.add_persistent('history', history)

    def _forward(self, adj_0, feature_0, adj_1, history_1, adj_sample_1,
                 history_sample_1, rf_sample_1):

        # Do not calculate CV because self.features are always fixed
        h = sparse_matmul2(adj_0, feature_0)
        if chainer.config.train:
            with chainer.no_backprop_mode():
                h.data = F.dropout(h.data, self.dropout).data
            h.eliminate_zeros()
        h = sparse_matmul2(h, self.W1)
        del feature_0, adj_0

        # Do NOT update history1 because it is a feature vector
        # self.history1[adjs[0][2]] = chainer.backends.cuda.to_cpu(h)
        h = F.relu(h)

        h1 = h
        h = sparse_matmul2(adj_sample_1, h - history_sample_1)
        # update history after using the old history
        self.history[rf_sample_1] = chainer.backends.cuda.to_cpu(h1.data)
        del h1, rf_sample_1, history_sample_1

        h += sparse_matmul2(adj_1, history_1)
        h = F.dropout(h, self.dropout)
        h = F.matmul(h, self.W2)

        return h

    def __call__(self, *args):
        mask = args[0]
        out = self._forward(*args[1:])

        loss = F.softmax_cross_entropy(out, self.labels[mask])
        accuracy = F.accuracy(out, self.labels[mask])

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        return loss

    def make_exact(self, converter, epochs, batchsize, device=None):
        """ Run forward propagation multiple times to get the exact histories.
        (Ref. section 4.1 in the paper)
        """
        indices = self.xp.random.permutation(self.features.shape[0])
        for _ in tqdm(range(epochs)):
            test_iter = chainer.iterators.SerialIterator(
                TupleDataset(indices), batch_size=batchsize, repeat=False,
                shuffle=False)
            for batch in test_iter:
                args = converter(batch, device=device)
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    self._forward(*args[1:])

    def evaluate(self, idx, converter, batchsize, device=None):
        outputs = []
        labels = []
        test_iter = chainer.iterators.SerialIterator(
            TupleDataset(idx), batch_size=batchsize, repeat=False,
            shuffle=False)
        for batch in test_iter:
            args = converter(batch, device=device)
            mask = args[0]
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                out = self._forward(*args[1:])
            outputs.append(chainer.backends.cuda.to_cpu(out.data))
            labels.append(chainer.backends.cuda.to_cpu(self.labels[mask]))
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        loss = F.softmax_cross_entropy(outputs, labels)
        accuracy = F.accuracy(outputs, labels)

        return float(loss.data), float(accuracy.data)

    def predict(self, *args):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            out = self._forward(*args[1:])
        pred = self.xp.argmax(out.data)
        return pred

    def predict_proba(self, *args):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            out = F.softmax(self._forward(*args[1:]))
        return out.data

    def to_gpu(self, device=None):
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(GCN, self).to_gpu(device=device)

    def to_cpu(self):
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(GCN, self).to_cpu()


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
