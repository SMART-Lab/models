# -*- coding: utf-8 -*-

import numpy as np
from blocks.bricks import Tanh, Linear
from blocks.initialization import IsotropicGaussian, Constant
from smartlearner import Trainer
from smartlearner.tasks.stopping_criteria import MaxEpochStopping
from smartlearner.tasks import tasks
from smartlearner.optimizers import SGD
from smartlearner.update_rules import ConstantLearningRate
from smartlearner.batch_scheduler import MiniBatchScheduler
from smartlearner.losses.reconstruction_losses import L2Distance

from smartmodels import ffnn_blocks
from projects.timeSeries import synthetic_dataset as dset
from smartmodels.utils import Timer

class LinearLayer(Linear):
    def __init__(self, input_dim, output_dim, **kwargs):
        if 'weights_init' not in kwargs.keys():
            kwargs['weights_init'] = IsotropicGaussian(0.1)
        if 'biases_init' not in kwargs.keys():
            kwargs['biases_init'] = Constant(0)
        super().__init__(input_dim, output_dim, **kwargs)
        self.initialize()


def train_sequence_ffnn():
    with Timer("Loading dataset"):
        synt_generator = dset.TimeSerieGenerator(1234, 8760)
        synt_generator.add_trend(3/8760)
        synt_generator.add_season(3, 24, 0)
        synt_generator.add_season(4, 8760, 240)
        synt_generator.add_noise(0.1)
        synt_generator.add_binary_covariate(0.1, 1, 1)

        trainset = chunking(dset.SyntheticDataset(synt_generator), 50)

        validset = None
        testset = None

    with Timer("Creating model"):
        output_size = 1
        blocks_topology = [LinearLayer(input_dim=trainset.input_size, output_dim=100),
                           Tanh(),
                           LinearLayer(input_dim=100, output_dim=150),
                           Tanh(),
                           LinearLayer(input_dim=150, output_dim=output_size)]
        model = ffnn_blocks.BlocksFFNN(trainset, blocks_topology)

    with Timer("Building optimizer"):
        optimizer = SGD(loss=L2Distance(model, trainset))
        optimizer.append_update_rule(ConstantLearningRate(0.001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(trainset, 128)

        trainer = Trainer(optimizer, batch_scheduler)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())
        trainer.append_task(MaxEpochStopping(3))

    with Timer("Training"):
        trainer.train()


def chunking(dset, chunk_length):
    np_dset = dset.inputs.get_value()
    dset_length = np_dset.shape[0]
    nb_chunks = dset_length - chunk_length

    chunks = np.expand_dims(np_dset[:chunk_length].ravel(), 0)
    targets = np.expand_dims(dset.targets.get_value()[chunk_length-1], 0) if dset.targets is not None else None

    for k in range(1, nb_chunks):
        new_chunk = np.expand_dims(np_dset[k:chunk_length+k].ravel(), 0)
        chunks = np.vstack((chunks, new_chunk))
        if targets is not None:
            targets = np.vstack((targets, np.expand_dims(dset.targets.get_value()[chunk_length+k-1], 0)))

    dset.inputs = chunks
    dset.targets = targets
    return dset


if __name__ == "__main__":
    train_sequence_ffnn()
