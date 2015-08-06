from math import sin, pi

import numpy as np
from numpy.random import RandomState
import theano

from smartlearner.interfaces.dataset import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, generator, generate_targets=True, name='SyntheticDataset', **kwargs):
        inputs, targets = generator.generate_dataset(generate_targets, **kwargs)
        super().__init__(inputs, targets, name)


class Generator(object):
    def __init__(self, seed, length):
        self.rng = RandomState(seed)
        self.length = length

    def generate_dataset(self, generate_target=True, starting_value=None, **kwargs):
        raise NotImplementedError("This method needs to be implemented in a specific subclass of Generator.")


class TimeSerieGenerator(Generator):
    def __init__(self, seed, length):
        super().__init__(seed, length)
        self._covariates = [] #[sigma, current_beta]
        self._effects = [] #list of step functions

    def add_effect(self, step_function):
        self._effects.append(step_function)

    def add_trend(self, speed):
        self.add_effect(lambda l, t: speed)

    def add_season(self, amplitude, period, starting_step):
        self.add_effect(lambda l, t: amplitude * (
                                     sin(2 * pi * t/period + starting_step * 2 * pi / period) -
                                     sin(2 * pi * (t-1)/period + starting_step * 2 * pi / period)))

    def add_cycle(self):
        pass

    def add_noise(self, sigma):
        self.add_effect(lambda l, t: self.rng.normal(0, sigma))

    def add_binary_covariate(self, probability, sigma_0, sigma):
        beta0 = self.rng.normal(0, sigma_0)
        self._covariates.append([probability, beta0, sigma])

    def generate_dataset(self, generate_target=True, starting_value=None, **kwargs):
        if generate_target:
            length = self.length + 1
        else:
            length = self.length

        if starting_value is None:
            starting_value = self.rng.lognormal(5, 2)

        inputs = np.zeros((length, len(self._covariates)+1))
        inputs[0, 0] = starting_value
        inputs[:, 1:] = self._generate_covariates(length)

        for t in range(1, length):
            last_val = inputs[t-1, 0]
            delta = 0

            for k, (_, _, sigma) in enumerate(self._covariates):
                if sigma > 0:
                    self._covariates[k][1] += self.rng.normal(0, sigma)
                delta += inputs[t-1, k] * self._covariates[k][1]

            for e in self._effects:
                delta += e(last_val, t)

            inputs[t, 0] = inputs[t-1, 0] + delta

        inputs = np.asarray(inputs, dtype=theano.config.floatX)

        if generate_target:
            targets = np.expand_dims(inputs[1:, 0], 1)
            inputs = inputs[:-1]
        else:
            targets = None

        return inputs, targets

    def _generate_covariates(self, length):
        return np.array(list(map(lambda p: self.rng.binomial(1, p, length), [c[0] for c in self._covariates]))).T
