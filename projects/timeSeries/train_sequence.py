# -*- coding: utf-8 -*-

import numpy as np

from smartmodels import ffnn
from projects.timeSeries import synthetic_dataset as dset
from smartmodels.utils import Timer, FullyConnectedLayer
from smartlearner import Trainer, tasks
from smartlearner.optimizers import SGD
from smartlearner.update_rules import ConstantLearningRate
from smartlearner.batch_scheduler import MiniBatchScheduler
from smartlearner.losses.reconstruction_losses import L2Distance


def train_sequence_ffnn():
    with Timer("Loading dataset"):
        synt_generator = dset.TimeSerieGenerator(1234, 8760*3)
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
        topology = [FullyConnectedLayer(100), FullyConnectedLayer(150)]
        model = ffnn.FFNN(trainset.input_size, output_size, topology, dropout_rate=0.5, use_batch_normalization=True)
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        optimizer = SGD(loss=L2Distance(model, trainset))
        optimizer.append_update_rule(ConstantLearningRate(0.001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(trainset, 128)
        stopping_criterion = tasks.MaxEpochStopping(3)

        trainer = Trainer(optimizer, batch_scheduler, stopping_criterion=stopping_criterion)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

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
