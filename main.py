import os

import constants
from data.StartingDataset import StartingDataset
# from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
from networks.LessDumbNetwork import AliceWithAGun

import torch

#TODO:
#dropout regularization
#unfreezing resnet

def main():

    model = AliceWithAGun()
    # model = StartingNetwork()

    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset()
    val_dataset = StartingDataset(eval=True)
    print(f"train size: {len(train_dataset)} - eval size: {len(val_dataset)}")

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
