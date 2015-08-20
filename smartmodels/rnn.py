import numpy as np
import theano.tensor as T
from theano import scan
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from smartlearner import Model, initializers as initer
from smartlearner.utils import sharedX


class RNN(Model):
    def __init__(self, input_size, output_size, hidden_layers, output_act_fct=lambda x: x, dropout_rate=0.0,
                 use_batch_normalization=False, seed=1234):
        self.tWs = []
        self.tbs = []
        self.tVs = []  # recurrent connections
        self.th0 = []  # Initial hidden layers
        self.act_fcts = []
        self.act_fcts_param = []
        self.batch_normalization_param = []

        self.dropout_rate = sharedX(dropout_rate, name='dropout_rate')
        self.use_dropout = sharedX(1.0 if dropout_rate > 0 else 0.0, name='use_dropout?')
        self.use_batch_normalization = use_batch_normalization
        self._trng = RandomStreams(seed)

        self._build_layers(input_size, hidden_layers, output_size, output_act_fct)

    def initialize(self,
                   w_initializer=initer.UniformInitializer(),
                   b_initializer=initer.ZerosInitializer(),
                   v_initializer=initer.UniformInitializer(),
                   h0_initializer=initer.ZerosInitializer()):
        for w, b in zip(self.tWs, self.tbs):
            w_initializer(w)
            b_initializer(b)
        for v, h0 in zip(self.tVs, self.th0):
            v_initializer(v)
            h0_initializer(h0)

    @property
    def parameters(self):
        return self.tWs + self.tbs + self.tVs + self.th0 +\
            self.act_fcts_param + self.batch_normalization_param

    def toggle_dropout(self):
        new_value = 0.0 if self.use_dropout.get_value() else 1.0
        self.use_dropout.set_value(new_value)
        return bool(new_value)

    def use_dropout(self, use_dropout):
        self.use_dropout.set_value(1.0 if use_dropout else 0.0)

    def get_model_output(self, X):
        def step_fct_generator(act_fct):
            def step_fct(input, past_layer, W, V, b):
                return act_fct(T.dot(input, W) + T.dot(past_layer, V) + b)
            return step_fct

        last_layer = X.dimshuffle(1, 0, 2)
        layer_number = 1
        updates = dict()

        for w, b, sigma, v, h0 in zip(self.tWs, self.tbs, self.act_fcts, self.tVs, self.th0):
            last_layer, updt = scan(fn=step_fct_generator(sigma),
                                    outputs_info=[T.alloc(h0, last_layer.shape[1], h0.shape[0])],
                                    sequences=[last_layer], non_sequences=[w, v, b])
            updates.update(updt)

            if self.dropout_rate.get_value():
                last_layer = self._dropout(last_layer)

            layer_number += 1

        # outputs aren't recurrent, needs to be done manually.
        last_layer = self.act_fcts[-1](T.dot(last_layer, self.tWs[-1]) + self.tbs[-1]).dimshuffle(1, 0, 2)

        return last_layer, updates

    def use_classification(self, X):
        pass

    def use_regression(self, X):
        return self.get_model_output(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _build_layers(self, input_size, hidden_layers, output_size, output_act_fct):
        last_layer_size = input_size

        for k, layer in enumerate(hidden_layers):
            self.tWs.append(sharedX(value=np.zeros((last_layer_size, layer.size)), name='W_'+str(k), borrow=True))
            self.tbs.append(sharedX(value=np.zeros((layer.size,)), name='b_'+str(k), borrow=True))
            self.tVs.append(sharedX(value=np.zeros((layer.size, layer.size)), name='V_'+str(k), borrow=True))
            self.th0.append(sharedX(value=np.zeros((layer.size,)), name='h0_'+str(k), borrow=True))
            self.act_fcts.append(layer.activation_function)
            self.act_fcts_param += layer.parameters
            last_layer_size = layer.size

        self.tWs.append(sharedX(value=np.zeros((last_layer_size, output_size)), name='W.out', borrow=True))
        self.tbs.append(sharedX(value=np.zeros((output_size,)), name='b.out', borrow=True))
        self.act_fcts.append(output_act_fct)

    def _dropout(self, layer):
        dpout_mask = ifelse(self.use_dropout,
                            self._trng.binomial(layer.shape, p=1-self.dropout_rate,
                                                n=1, dtype=layer.dtype),
                            T.fill(layer, 1-self.dropout_rate))

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
