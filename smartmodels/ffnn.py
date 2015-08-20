import theano.tensor as T
from smartlearner import Model
from blocks.graph import ComputationGraph
from blocks.bricks import Linear


class FFNN(Model):
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 seed=1234):
        self._input_dim = input_dim
        self._trng = T.shared_randomstreams.RandomStreams(seed)
        self.topology = hidden_layers
        self._graph = None

    @property
    def parameters(self):
        if self._graph is None:
            raise NotImplementedError("You need to use the model in a loss before calling this property.")
        return ComputationGraph(self._graph).parameters

    def get_output(self, inputs):
        if self._graph is None:
            self._graph = self._build_layers(inputs)
        return self._graph

    @property
    def updates(self):
        return {}

    def use_classification(self, inputs):
        probs = self.get_output(inputs)
        return T.argmax(probs, axis=1, keepdims=True)

    def use_regression(self, inputs):
        return self.get_output(inputs)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _build_layers(self, inputs):
        last_layer = inputs
        last_size = self._input_dim

        for layer in self.topology:
            activation = Linear(input_dim=last_size, output_dim=layer.size,
                                weights_init=layer.w_init, biases_init=layer.b_init)
            act_fct = layer.activation_function

            activation.initialize()
            act_fct.initialize()

            x = activation.apply(last_layer)
            last_layer = act_fct.apply(x)

            last_size = layer.size

        return last_layer
