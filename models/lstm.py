from itertools import chain
#3.5 from typing import Callable, Iterable, Tuple

import numpy as np
import theano.tensor as T
from theano import scan, Variable
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import sigmoid

from smartpy import Model
from smartpy.misc.utils import sharedX
from .utils import WeightsInitializer as WI, FullyConnectedLayer


class LSTM(Model):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: Iterable[FullyConnectedLayer],
                 output_act_fct: Callable[[np.ndarray], np.ndarray]=lambda x: x,
                 dropout_rate: float=0.0,
                 use_batch_normalization: bool=False,
                 seed: int=1234):
        self.layer_parameters = []  #{Wo, Wf, Wi, Wc, Uo, Uf, Ui, Uc, Vo, Vf, Vi, bo, bf, bi, c0, h0}
        self.output_parameters = {'W_out': None, 'b_out': None}

        self.act_fcts = []
        self.act_fcts_param = []
        self.batch_normalization_param = []

        self.dropout_rate = sharedX(dropout_rate, name='dropout_rate')
        self.use_dropout = sharedX(1.0 if dropout_rate > 0 else 0.0, name='use_dropout?')
        self.use_batch_normalization = use_batch_normalization
        self._trng = RandomStreams(seed)

        self._build_layers(input_size, hidden_layers, output_size, output_act_fct)

    def initialize(self,
                   wu_initializer: Callable[[Tuple(int, int)], np.ndarray]=WI().uniform,
                   bch_initializer: Callable[[Tuple(int, int)], np.ndarray]=WI().zeros,
                   v_initializer: Callable[[Tuple(int, int)], np.ndarray]=WI().uniform):
        for d in self.layer_parameters:
            for name in ['Wo', 'Wf', 'Wi', 'Wc', 'Uo', 'Uf', 'Ui', 'Uc']:
                array = d[name]
                array.set_value(wu_initializer(array.get_value().shape))
            for name in ['bo', 'bf', 'bi', 'c0', 'h0']:
                array = d[name]
                array.set_value(bch_initializer(array.get_value().shape))
            for name in ['Vo', 'Vf', 'Vi']:
                array = d[name]
                array.set_value(v_initializer(array.get_value().shape))

        array = self.output_parameters['W_out']
        array.set_value(wu_initializer(array.get_value().shape))
        array = self.output_parameters['b_out']
        array.set_value(bch_initializer(array.get_value().shape))

    @property
    def parameters(self):
        return  list(chain.from_iterable([x.keys() for x in self.layer_parameters])+\
            self.act_fcts_param + self.batch_normalization_param

    def toggle_dropout(self):
        new_value = 0.0 if self.use_dropout.get_value() else 1.0
        self.use_dropout.set_value(new_value)
        return bool(new_value)

    def use_dropout(self, use_dropout: bool):
        self.use_dropout.set_value(1.0 if use_dropout else 0.0)

    def get_model_output(self, X: Variable):
        def step_fct_generator(act_fct):
            def step_fct(input, past_layer, past_memory,
                         wo, wf, wi, wc,
                         uo, uf, ui, uc,
                         vo, vf, vi,
                         bo, bf, bi):
                f = sigmoid(T.dot(wf, input) + T.dot(uf, past_layer) + T.dot(vf, past_memory) + bf)
                i = sigmoid(T.dot(wi, input) + T.dot(ui, past_layer) + T.dot(vi, past_memory) + bi)
                o = sigmoid(T.dot(wo, input) + T.dot(uo, past_layer) + T.dot(vo, past_memory) + bo)
                c = f * past_memory + i * act_fct(T.dot(wc, input) + T.dot(uc, past_layer))
                h = o * act_fct(c)
                return h, c
            return step_fct

        last_layer = X
        layer_number = 1
        updates = dict()

        for d, sigma in zip(self.layer_parameters, self.act_fcts):
            last_layer, updt = scan(fn=step_fct_generator(sigma),
                                    outputs_info=[d['h0'], d['c0']],
                                    sequences=[last_layer],
                                    non_sequences=[d['Wo'], d['Wf'], d['Wi'], d['Wc'],
                                                   d['Uo'], d['Uf'], d['Ui'], d['Uc'],
                                                   d['Vo'], d['Vf'], d['Vi'],
                                                   d['bo'], d['bf'], d['bi']])
            updates.update(updt)

            if not self.dropout_rate.get_value():
                last_layer = self._dropout(last_layer)

            layer_number += 1

        # outputs aren't recurrent, needs to be done manually.
        last_layer = self.act_fcts[-1](T.dot(self.output_parameters['W_out'], last_layer) +
                                       self.output_parameters['b_out'])  # check shapes matches

        return last_layer, updates

    def use_classification(self, X):
        pass

    def use_regression(self, X):
        return self.get_model_output(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _build_layers(self,
                      input_size: int,
                      hidden_layers: Iterable[FullyConnectedLayer],
                      output_size: int,
                      output_act_fct: Callable[[np.ndarray], np.ndarray]):
        last_layer_size = input_size

        for k, layer in enumerate(hidden_layers):
            d = dict()

            W = (last_layer_size, layer.size)
            U = (layer.size, layer.size)
            V = (layer.size,)

            names_and_shapes = zip(['Wo', 'Wf', 'Wi', 'Wc', 'Uo', 'Uf', 'Ui', 'Uc', 'Vo', 'Vf', 'Vi', 'bo', 'bf', 'bi', 'c0', 'h0'],
                                   [W, W, W, W, U, U, U, U, V, V, V, V, V, V, V, V, V])
            for name, shape in names_and_shapes:
                tname = name + '_' + str(k)
                d[name] = sharedX(value=np.zeros(shape), name=tname)

            self.act_fcts.append(layer.activation_function)
            self.act_fcts_param += layer.parameters
            last_layer_size = layer.size

        self.output_parameters['W_out'] = sharedX(value=np.zeros((last_layer_size, output_size)), name='W.out')
        self.output_parameters['b_out'] = sharedX(value=np.zeros((output_size,)), name='b.out')
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

