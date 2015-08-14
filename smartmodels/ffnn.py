import theano.tensor as T
from smartlearner import Model
from blocks.graph import ComputationGraph
from blocks.bricks import Linear


class FFNN(Model):
    def __init__(self,
                 trainset,
                 hidden_layers,
                 seed=1234):
        self._trng = T.shared_randomstreams.RandomStreams(seed)
        self._graph = self._build_layers(trainset, hidden_layers)

    @property
    def parameters(self):
        return ComputationGraph(self.output[0]).parameters

    @property
    def output(self):
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

    def _build_layers(self, input_dataset, topology):
        last_layer = input_dataset.symb_inputs
        last_size = input_dataset.input_size

        for layer in topology:
            activation = Linear(input_dim=last_size, output_dim=layer.size,
                                weights_init=layer.w_init, biases_init=layer.b_init)
            act_fct = layer.activation_function

            last_layer = act_fct.apply(activation.apply(last_layer))
            last_size = layer.size

            activation.initialize()
            act_fct.initialize()

        return last_layer
