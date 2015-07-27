import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse

from smartpy import Model
from smartpy.misc.utils import sharedX
from .utils import WeightsInitializer


class FFNN(Model):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers,
                 output_act_fct=lambda x: x,
                 dropout_rate=0.0,
                 use_batch_normalization=False,
                 seed=1234):
        self.tWs = []
        self.tbs = []
        self.act_fcts = []
        self.act_fcts_param = []
        self.batch_normalization_param = []
        self.dropout_rate = sharedX(dropout_rate, name='dropout_rate')
        self.use_dropout = sharedX(1.0 if dropout_rate > 0 else 0.0, name='use_dropout?')
        self.use_batch_normalization = use_batch_normalization
        self._trng = T.shared_randomstreams.RandomStream(seed)

        self._build_layers(input_size, hidden_layers, output_size, output_act_fct)

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
        return self.tWs + self.tbs + self.act_fcts_param + self.batch_normalization_param

    def toggle_dropout(self):
        new_value = 0.0 if self.use_dropout.get_value() else 1.0
        self.use_dropout.set_value(new_value)
        return bool(new_value)

    def use_dropout(self, use_dropout):
        self.use_dropout.set_value(1.0 if use_dropout else 0.0)

    def get_model_output(self, X):
        last_layer = X
        layer_number = 1
        for w, b, sigma in zip(self.tWs, self.tbs, self.act_fcts):
            if self.use_batch_normalization:
                last_layer = self._batch_normalization(last_layer, str(layer_number))

            last_layer = sigma(T.dot(w, last_layer) + b)

            if not self.dropout_rate.get_value():
                last_layer = self._dropout(last_layer)

            layer_number += 1

        return last_layer

    def use(self, X):
        probs = self.get_model_output(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _build_layers(self, input_size, hidden_layers, output_size, output_act_fct):
        last_layer_size = input_size

        for k, layer in enumerate(hidden_layers):
            self.tWs.append(sharedX(value=np.zeros((last_layer_size, layer.size)), name='W'+str(k), borrow=True))
            self.tbs.append(sharedX(value=np.zeros((layer.size,)), name='b'+str(k), borrow=True))
            self.act_fcts.append(layer.activation_function)
            self.act_fcts_param += layer.parameters

        self.tWs.append(sharedX(value=np.zeros((last_layer_size, output_size)), name='W.out', borrow=True))
        self.tbs.append(sharedX(value=np.zeros((output_size,)), name='b.out', borrow=True))
        self.act_fcts.append(output_act_fct)

    def _dropout(self, layer):
        dpout_mask = ifelse(self.use_dropout,
                            self._trng.binomial(layer.shape, 1-self.dropout_rate,
                                                n=1, dtype=layer.dtype) * layer,
                            1-self.dropout_rate)

        layer = ifelse(self.dropout_rate > 0,
                       layer * dpout_mask,
                       layer)
        return layer

    def _batch_normalization(self, activation, name_prefix='', eps=1e-6):
        gamma = sharedX(1, name=name_prefix + '_gamma')
        beta = sharedX(0, name=name_prefix + '_beta')
        self.batch_normalization_param += [gamma, beta]

        mu = T.sum(activation)/activation.shape[0]
        sig2 = T.sum(T.sqr(activation - mu))/activation.shape[0]
        x_hat = (activation - mu)/T.sqrt(eps + sig2)

        return gamma * x_hat + beta