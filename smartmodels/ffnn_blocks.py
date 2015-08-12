from functools import reduce

import theano.tensor as T
from smartlearner import Model
from blocks.graph import ComputationGraph


class BlocksFFNN(Model):
    def __init__(self,
                 trainset,
                 hidden_layers,
                 seed=1234):
        self._trng = T.shared_randomstreams.RandomStreams(seed)
        self._graph = None
        self.layers = hidden_layers
        self.input = trainset.symb_inputs

    @property
    def parameters(self):
        return ComputationGraph(self.output[0]).parameters

    @property
    def output(self):
        if self._graph is None:
            self._graph = reduce(lambda x, f: f.apply(x), self.layers, self.input)

        return self._graph, {}

    def use_classification(self):
        probs, _ = self.output
        return T.argmax(probs, axis=1, keepdims=True)

    def use_regression(self):
        return self.output[0]

    def save(self, path):
        pass

    def load(self, path):
        pass

