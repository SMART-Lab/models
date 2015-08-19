#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smartmodels.perceptron import Perceptron
from smartmodels.utils import load_mnist
from smartmodels.utils import Timer

from smartlearner import Trainer, tasks
from smartlearner import stopping_criteria, views
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.losses.classification_losses import NegativeLogLikelihood as NLL
from smartlearner.batch_schedulers import MiniBatchScheduler


def test_simple_perceptron():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_mnist()

    with Timer("Creating model"):
        # TODO: We should the number of different targets in the dataset,
        #       but I'm not sure how to do it right (keep in mind the regression?).
        output_size = 10
        model = Perceptron(trainset.input_size, output_size)
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        optimizer = SGD(loss=NLL(model, trainset))
        optimizer.append_direction_modifier(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(trainset, 100)

        trainer = Trainer(optimizer, batch_scheduler)
        trainer.append_task(stopping_criteria.MaxEpochStopping(10))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        classif_error = views.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    with Timer("Training"):
        trainer.train()
