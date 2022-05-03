import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from networks.DumbNetwork import CNN
from networks.StartingNetwork import StartingNetwork


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model = StartingNetwork()
    # model = CNN()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        losses = []
        model.train() #tells your model that you are training the model
        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            batch_inputs, batch_labels = batch
            optimizer.zero_grad() #PyTorch accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters, and the newly-computed gradient
            batch_outputs = model(batch_inputs)
            # print(type(batch_labels))
            # print(batch_labels)
            loss = loss_fn(batch_outputs, batch_labels.clone().detach())
            batch_outputs = batch_outputs.argmax(axis=1)
            # print(batch_outputs)
            loss.backward()
            losses.append(loss)
            optimizer.step()


            # Periodically evaluate our model + log to Tensorboard
            #training data gives model actual weights
            #validataion data is data the model hasn't seen yet ("real world") to better represent how the model does realistically
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard. ???
                training_accuracy = compute_accuracy(batch_outputs, batch_labels)
                print()
                print(f"Training Results: Accuracy - {training_accuracy * 100}%, Loss - {sum(losses) / len(losses)}")

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)

            step += 1

        # print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs.float()) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval() #this disables gradient computation automagically
    accuracies = []
    losses = []
    for batch in val_loader:
        images, labels = batch
        outputs = model(images)  # axis does squishes the 2d tensor of probability vectors into 1d tensor
        outputs = outputs.to(torch.float)
        labels = labels.to(torch.long)
        # print(type(outputs))
        # print(labels)
        # print("Output size:")
        losses.append(loss_fn(outputs, labels))
        outputs = outputs.argmax(axis=1)
        accuracies.append(compute_accuracy(outputs, labels) * 100)
    print(f"Evaluation Results: Accuracy - {sum(accuracies) / len(accuracies)}%, Loss - {sum(losses) / len(losses)}")
        
