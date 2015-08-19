# -*- coding: utf-8 -*-

import numpy as np

from smartlearner import Trainer
from smartlearner.stopping_criteria import MaxEpochStopping, EarlyStopping
from smartlearner import tasks, views
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import MiniBatchScheduler
from smartlearner.losses.reconstruction_losses import L2Distance
from smartlearner.utils import split_dataset
from smartmodels import ffnn
from projects.timeSeries import synthetic_dataset as dset
from smartmodels.utils import Timer, FullyConnectedLayer, OutputLayer


def train_sequence_ffnn():
    with Timer("Loading dataset"):
        synt_generator = dset.TimeSerieGenerator(1234, 8760)
        synt_generator.add_trend(3/8760)
        synt_generator.add_season(3, 24, 0)
        synt_generator.add_season(4, 8760, 240)
        synt_generator.add_noise(0.1)
        synt_generator.add_binary_covariate(0.1, 0.01, 0.0001)

        synth_datasets = split_dataset(dset.SyntheticDataset(synt_generator), [7, 2, 1])

        trainset = chunking(synth_datasets[0], 50)
        validset = chunking(synth_datasets[1], 50)
        testset = chunking(synth_datasets[2], 50)

    with Timer("Creating model"):
        output_size = 1
        topology = [FullyConnectedLayer(10), FullyConnectedLayer(15), OutputLayer(output_size)]
        model = ffnn.FFNN(trainset.input_shape[-1], topology)

    with Timer("Building optimizer"):
        optimizer = SGD(loss=L2Distance(model, trainset))
        optimizer.append_direction_modifier(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(trainset, 128)

        trainer = Trainer(optimizer, batch_scheduler)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        l2_error = views.ReconstructionError(model.use_regression, trainset)
        trainer.append_task(tasks.Print("Trainset - Reconstruction error: {0:.3f} ± {1:.3f}", l2_error.mean, l2_error.stderror))

        l2_error = views.ReconstructionError(model.use_regression, validset)
        trainer.append_task(tasks.Print("Validset - Reconstruction error: {0:.3f} ± {1:.3f}", l2_error.mean, l2_error.stderror))

        trainer.append_task(MaxEpochStopping(20000))
        trainer.append_task(EarlyStopping(l2_error.mean, 100))

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
