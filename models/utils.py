from __future__ import print_function

# MLPython datasets wrapper
import os
import sys
import json
import theano
import theano.sandbox.softsign
import numpy as np
from time import time

from smartpy import Dataset

DATASETS_ENV = 'DATASETS'


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


class WeightsInitializer(object):
    def __init__(self, random_seed=None):
        self.rng = np.random.mtrand.RandomState(random_seed)

    @staticmethod
    def default(proposed_instance, default_instance):
        if proposed_instance is None:
            return default_instance
        else:
            return proposed_instance

    def _init_range(self, dim):
        return np.sqrt(6. / (dim[0] + dim[1]))

    def uniform(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, high=init_range, size=dim), dtype=theano.config.floatX)

    def zeros(self, dim):
        return np.zeros(dim, dtype=theano.config.floatX)

    def diagonal(self, dim):
        W_values = self.zeros(dim)
        np.fill_diagonal(W_values, 1)
        return W_values

    def orthogonal(self, dim):
        max_dim = max(dim)
        return np.linalg.svd(self.uniform((max_dim, max_dim)))[2][:dim[0], :dim[1]]

    def gaussian(self, dim):
        return np.asarray(self.rng.normal(loc=0, scale=self._init_range(dim), size=dim), dtype=theano.config.floatX)


def load_mnist():
    #Temporary patch until we build the dataset manager
    dataset_name = "mnist"

    datasets_repo = os.environ.get(DATASETS_ENV, './datasets')
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.mkdir(repo)

            import urllib.request
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', os.path.join(repo, 'mnist_train.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', os.path.join(repo, 'mnist_valid.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', os.path.join(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [os.path.join(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]

        def parse_file(filename):
            return np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        trainset_inputs, trainset_targets = trainset[:, :-1], trainset[:, [-1]]
        validset_inputs, validset_targets = validset[:, :-1], validset[:, [-1]]
        testset_inputs, testset_targets = testset[:, :-1], testset[:, [-1]]

        np.savez(dataset_npy,
                 trainset_inputs=trainset_inputs, trainset_targets=trainset_targets,
                 validset_inputs=validset_inputs, validset_targets=validset_targets,
                 testset_inputs=testset_inputs, testset_targets=testset_targets)

    data = np.load(dataset_npy)
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), data['trainset_targets'].astype(theano.config.floatX))
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), data['validset_targets'].astype(theano.config.floatX))
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), data['testset_targets'].astype(theano.config.floatX))

    return trainset, validset, testset


class FullyConnectedLayer(object):
    def __init__(self, size, activation_function='tanh'):
        self.size = size
        self._parameters = []
        self.activation_function = self._choose_act_fct(activation_function)

    @property
    def parameters(self):
        return self._parameters

    def _choose_act_fct(self, act_fct_str):
        if act_fct_str == 'sigmoid':
            act_fct = theano.tensor.nnet.sigmoid
        elif act_fct_str == 'relu':
            def relu(x):
                return theano.tensor.maximum(x, 0.0)
            act_fct = relu
        elif act_fct_str == 'softplus':
            act_fct = theano.tensor.nnet.softplus
        elif act_fct_str == 'tanh':
            act_fct = theano.tensor.tanh
        elif act_fct_str == 'softsign':
            act_fct = theano.sandbox.softsign.softsign
        elif act_fct_str == 'brain':
            def brain(x):
                return theano.tensor.maximum(theano.tensor.log(theano.tensor.maximum(x + 1, 1)), 0.0)
            act_fct = brain
        else:
            raise ValueError(act_fct_str + " is not a valid activation function name.")

        return act_fct


class LSTMLayer(FullyConnectedLayer):
    def __init__(self, size, memory_size, activation_function='tanh'):
        super().__init__(size, activation_function)
        self.memory_size = memory_size
