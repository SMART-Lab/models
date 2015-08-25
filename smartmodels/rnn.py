import numpy as np
from blocks.bricks import Linear
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.initialization import Identity
from blocks.roles import PARAMETER, INITIAL_STATE, add_role
from smartlearner.utils import sharedX

from .ffnn import FFNN


class RNN(FFNN):
    def use_classification(self, inputs):
        pass

    def _build_layers(self, inputs):
        last_layer_size = self._input_dim
        last_layer = inputs

        for layer in self.topology:
            activation = Linear(last_layer_size, layer.size, weights_init=layer.w_init, biases_init=layer.b_init)
            recurrent_step = SimpleRecurrent(layer.size, layer.activation_function, weights_init=Identity())
            h0 = sharedX(np.zeros((layer.size,)), name='h0')
            add_role(h0, PARAMETER)
            add_role(h0, INITIAL_STATE)

            activation.initialize()
            recurrent_step.initialize()

            last_layer = recurrent_step.apply(inputs=activation.apply(last_layer), states=h0, iterate=False)
            last_layer_size = layer.size

        return last_layer
