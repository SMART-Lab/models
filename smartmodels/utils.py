from __future__ import print_function

# MLPython datasets wrapper
import os
import sys
import json
from time import time

import theano
import theano.sandbox.softsign
import numpy as np
from smartlearner import Dataset
from blocks.bricks import Tanh, Logistic, Rectifier, Softplus, Identity, Softmax
from blocks.initialization import IsotropicGaussian, Constant

DATASETS_ENV = 'DATASETS'


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


class Timer:
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


def load_mnist():
    #Temporary patch until we build the dataset manager
    dataset_name = "mnist"

    datasets_repo = os.environ.get(DATASETS_ENV, os.path.join(os.environ["HOME"], '.smartdatasets'))
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


class FullyConnectedLayer:
    def __init__(self, size, activation_function='tanh', w_init=IsotropicGaussian(0.1), b_init=Constant(0)):
        self.size = size
        self.activation_function = self._choose_act_fct(activation_function)
        self.w_init = w_init
        self.b_init = b_init

    def _choose_act_fct(self, act_fct_str):
        if act_fct_str in ['sigmoid', 'sigm']:
            act_fct = Logistic()
        elif act_fct_str == 'relu':
            act_fct = Rectifier()
        elif act_fct_str == 'softplus':
            act_fct = Softplus()
        elif act_fct_str == 'tanh':
            act_fct = Tanh()
        elif act_fct_str == 'softsign':
            raise NotImplementedError("No softsign bricks exists yet.")
            # act_fct = theano.sandbox.softsign.softsign
        elif act_fct_str == 'brain':
            raise NotImplementedError("No softsign bricks exists yet.")
            # def brain(x):
            #     return theano.tensor.maximum(theano.tensor.log(theano.tensor.maximum(x + 1, 1)), 0.0)
            # act_fct = brain
        elif act_fct_str == 'linear':
            act_fct = Identity()
        elif act_fct_str == 'softmax':
            act_fct = Softmax()
        else:
            raise ValueError(act_fct_str + " is not a valid activation function name.")

        return act_fct


class OutputLayer(FullyConnectedLayer):
    def __init__(self, output_size, activation_function='linear', w_init=IsotropicGaussian(0.1), b_init=Constant(0)):
        super().__init__(output_size, activation_function, w_init, b_init)
