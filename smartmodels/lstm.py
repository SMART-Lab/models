import numpy as np
from blocks.bricks import Linear
from blocks.bricks.recurrent import LSTM as LSTM_brick
from blocks.initialization import Uniform
from blocks.roles import PARAMETER, INITIAL_STATE, add_role
from smartlearner.utils import sharedX

from smartmodels.utils import FullyConnectedLayer, OutputLayer

from .ffnn import FFNN


class LSTM(FFNN):
    def use_classification(self, inputs):
        pass

    def _build_layers(self, inputs):
        last_layer_size = self._input_dim
        last_layer = inputs

        for layer in self.topology:
            if isinstance(layer, OutputLayer):
                activation = Linear(last_layer_size, layer.size, weights_init=layer.w_init, biases_init=layer.b_init)
                activation.initialize()
                last_layer = layer.activation_function.apply(activation.apply(last_layer))

            elif isinstance(layer, FullyConnectedLayer):
                activation = Linear(last_layer_size, layer.size * 4, weights_init=layer.w_init, biases_init=layer.b_init)
                recurrent_step = LSTM_brick(layer.size, layer.activation_function, weights_init=Uniform(width=0.4))
                h0 = sharedX(np.zeros((layer.size,)), name='h0')
                add_role(h0, PARAMETER)
                add_role(h0, INITIAL_STATE)

                c0 = sharedX(np.zeros((layer.size,)), name='h0')
                add_role(c0, PARAMETER)
                add_role(c0, INITIAL_STATE)

                activation.initialize()
                recurrent_step.initialize()

                last_layer = recurrent_step.apply(inputs=activation.apply(last_layer),
                                                  states=h0,
                                                  cells=c0,
                                                  iterate=False)
                # A LSTM returns (next_state, next_cell)
                last_layer = last_layer[0]

            else:
                raise ValueError("{} should be an instance of OutputLayer or FullyConnectedLayer".format(layer))

            last_layer_size = layer.size

        return last_layer
