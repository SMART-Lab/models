import numpy as np

from smartmodels import ffnn
from projects.timeSeries.run_script import *

def train_sequence_ffnn():
    synth_datasets = create_synthetic_datasets()

    trainset = chunking(synth_datasets[0], 50)
    validset = chunking(synth_datasets[1], 50)
    testset = chunking(synth_datasets[2], 50)

    model = create_model(ffnn.FFNN, trainset)
    optimizer = create_optimizer(model, trainset)
    trainer = create_trainer(model, optimizer, trainset, validset)
    training(trainer)


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
