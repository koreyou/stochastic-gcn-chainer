**This project is still WIP.**

# Chainer implementation of Stochastic GCN

This project implements GCN ([Chen et al. 2018. Stochastic Training of Graph Convolutional Networks with Variance Reduction. NIPS.](https://arxiv.org/abs/1609.02907)) and GAT ([Veličković et al. 2018. Graph Attention Networks. ICLR.](http://proceedings.mlr.press/v80/chen18p.html)) with [Chainer](https://chainer.org/).
The project includes codes to reproduce the experiments on multiple graph classification datasets. **This is NOT an official implementation by the authors.**

I referenced [@takyamamoto's implementation of GCN](https://github.com/takyamamoto/Graph-Convolution-Chainer) to implement this project.

# How to Run

## Prerequisite

I have only tested the code on Python 3.6.4. Install dependent library by running:

```
pip install -r requirements.txt
```

You need to install `cupy` to enable GPU.

## Running training and evaluation

Run:

```
python train.py
```

Refer to `python train.py -h` for the options.


# Reproducing the paper

I get around 81-83% test accuracy as in the original GCN.
I will add more formal experiment result later.

# TODO

Things to do before removing "WIP" flag:

* More efficient adjacency matrix generation.
* Add experiment results.

# Licensing

`load_data` module and all files under `data/` directory have been derived from [Dr. Kipf's repository](https://github.com/tkipf/gcn/tree/98357bded82fdc19595aa5b1448ee0e76557a399), so refer to the original repository for licensing.
Other files are distributed under [CC0](./LICENSE).
