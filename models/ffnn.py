import numpy as np
import theano.tensor as T

from smartpy import Model
from smartpy.misc.utils import sharedX
from .utils import WeightsInitializer


class FFNN(Model):
    def __init__(self, input_size, output_size, hidden_layers, output_act_fct=lambda x: x):
        self.tWs = []
        self.tbs = []
        self.act_fcts = []
        self.act_fcts_param = []
        last_layer_size = input_size

        for k, layer in enumerate(hidden_layers):
            self.tWs.append(sharedX(value=np.zeros((last_layer_size, layer.size)), name='W'+str(k), borrow=True))
            self.tbs.append(sharedX(value=np.zeros((layer.size,)), name='b'+str(k), borrow=True))
            self.act_fcts.append(layer.activation_function)
            self.act_fcts_param += layer.parameters

        self.tWs.append(sharedX(value=np.zeros((last_layer_size, output_size)), name='W.out', borrow=True))
        self.tbs.append(sharedX(value=np.zeros((output_size,)), name='b.out', borrow=True))
        self.act_fcts.append(output_act_fct)

    def initialize(self, w_initializer=None, b_initializer=None):
        if w_initializer is None:
            w_initializer = WeightsInitializer().uniform
        if b_initializer is None:
            b_initializer = WeightsInitializer().zeros

        for w in self.tWs:
            w.set_value(w_initializer(w.get_value().shape))
        for b in self.tbs:
            b.set_value(b_initializer(b.get_value().shape))

    @property
    def parameters(self):
        return self.tWs + self.tbs + self.act_fcts_param

    def get_model_output(self, X):
        last_layer = X
        for w, b, sigma in zip(self.tWs, self.tbs, self.act_fcts):
            last_layer = sigma(T.dot(w, last_layer) + b)
        return last_layer

    def use(self, X):
        probs = self.get_model_output(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        pass

    def load(self, path):
        pass