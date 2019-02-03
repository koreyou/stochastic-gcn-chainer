import argparse
import os

import chainer
from chainer import training
from chainer.datasets import TupleDataset
from chainer.training import extensions
import numpy as np
import scipy.sparse as sp

from nets import GCN
from graphs import load_data
from sampling import random_sampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', '-m', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'pubmed', 'citeseer'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=5000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=32,
                        help='Number of units')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--validation-interval', type=int, default=1,
                        help='Number of updates before running validation')
    parser.add_argument('--normalization', default='gcn',
                        choices=['pygcn', 'gcn'],
                        help='Variant of adjacency matrix normalization method to use')
    args = parser.parse_args()

    print("Loading data")
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset, normalization=args.normalization)

    train_iter = chainer.iterators.SerialIterator(
        TupleDataset(idx_train), batch_size=args.batchsize, shuffle=False)
    dev_iter = chainer.iterators.SerialIterator(
        TupleDataset(idx_val), batch_size=args.batchsize, repeat=False, shuffle=False)

    # Set up a neural network to train.
    model = GCN(adj, features, labels, args.unit, dropout=args.dropout)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    if args.weight_decay > 0.:
        optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(args.weight_decay))

    if args.resume != None:
        print("Loading model from " + args.resume)
        chainer.serializers.load_npz(args.resume, model)

    converter = create_convert_func(model, adj, 2)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=converter, device=args.gpu)
    trigger = training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', patients=100,
        check_trigger=(args.validation_interval, 'epoch'),
        max_trigger=(args.epoch, 'epoch'))
    trainer = training.Trainer(updater, trigger, out=args.out)

    trainer.extend(
        extensions.Evaluator(dev_iter, model, converter=converter, device=args.gpu),
        trigger=(args.validation_interval, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Take a best snapshot
    record_trigger = training.triggers.MinValueTrigger(
        'validation/main/loss', (args.validation_interval, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(model, 'best_model.npz'),
        trigger=record_trigger)

    trainer.run()

    chainer.serializers.load_npz(
        os.path.join(args.out, 'best_model.npz'), model)

    print('Updating history for the test nodes...')
    model.make_exact(converter, 10, args.batchsize, device=args.gpu)

    chainer.serializers.save_npz(
        os.path.join(args.out, 'best_model.npz'), model)

    print('Running test...')
    _, accuracy = model.evaluate(
        idx_test, converter, args.batchsize, device=args.gpu)
    print('Test accuracy = %f' % accuracy)


def create_convert_func(model, adj, n_samples):
    def convert(batch, device=None, with_label=True):
        idx = np.array([i for i, in batch], dtype=np.int32)
        mask = np.zeros([adj.shape[0]], dtype=bool)
        mask[idx] = True
        adjs_sample, adjs, rfs_sample, rfs = random_sampling(
            adj, mask, 2, n_samples)

        adj_0 = adjs[0]
        feature_0 = model.features[rfs[0]]
        adj_1 = adjs[1]
        history_1 = model.history[rfs[1]]
        adj_sample_1 = adjs_sample[1]
        history_sample_1 = model.history[rfs_sample[1]]
        rf_sample_1 = rfs_sample[1]

        batch = [
            mask, adj_0, feature_0, adj_1, history_1, adj_sample_1,
            history_sample_1, rf_sample_1
        ]
        if device is not None and device >= 0:
            batch = [to_gpu(x, device) for x in batch]
        return tuple(batch)
    return convert


def to_gpu(x, device):
    if sp.issparse(x):
        from cupyx.scipy.sparse import csr_matrix as xcsr_matrix
        return xcsr_matrix(x)
    else:
        return chainer.dataset.to_device(device, x)


if __name__ == '__main__':
    main()