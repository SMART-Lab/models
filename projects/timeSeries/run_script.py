from smartlearner import Trainer
from smartlearner.stopping_criteria import MaxEpochStopping, EarlyStopping
from smartlearner import tasks, views
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import MiniBatchScheduler
from smartlearner.losses.reconstruction_losses import L2Distance
from smartlearner.utils import split_dataset
from projects.timeSeries import synthetic_dataset as dset
from smartmodels.utils import Timer, FullyConnectedLayer, OutputLayer

def create_synthetic_datasets():
    with Timer("Building dataset"):
        synt_generator = dset.TimeSerieGenerator(1234, 8760)
        synt_generator.add_trend(3/8760)
        synt_generator.add_season(3, 24, 0)
        synt_generator.add_season(4, 8760, 240)
        synt_generator.add_noise(0.1)
        synt_generator.add_binary_covariate(0.1, 0.01, 0.0001)

        datasets = split_dataset(dset.SyntheticDataset(synt_generator), [7, 2, 1])
    return datasets


def create_model(model_class, training_set):
    with Timer("Creating model"):
        output_size = 1
        topology = [FullyConnectedLayer(10), FullyConnectedLayer(15), OutputLayer(output_size)]
        model = model_class(training_set.input_shape[-1], topology)
    return model


def create_optimizer(model, training_set):
    with Timer("Building optimizer"):
        optimizer = SGD(loss=L2Distance(model, training_set))
        optimizer.append_direction_modifier(ConstantLearningRate(0.0001))
    return optimizer


def create_trainer(model, optimizer, training_set, validation_set):
    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(training_set, 128)

        trainer = Trainer(optimizer, batch_scheduler)

        # Print time for one epoch
        #trainer.append_task(tasks.PrintEpochDuration())
        #trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        l2_error = views.RegressionError(model.use_regression, training_set)
        #trainer.append_task(tasks.Print("Trainset - Regression error: {0:.3f} ± {1:.3f}", l2_error.mean, l2_error.stderror))

        l2_error = views.RegressionError(model.use_regression, validation_set)
        #trainer.append_task(tasks.Print("Validset - Regression error: {0:.3f} ± {1:.3f}", l2_error.mean, l2_error.stderror))

        trainer.append_task(MaxEpochStopping(20000))
        trainer.append_task(EarlyStopping(l2_error.mean, 100))
    return trainer


def training(trainer):
    with Timer("Training"):
        trainer.train()
