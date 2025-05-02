import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from helpers import accuracy_fn


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = "mps"
               ):
    """
    Trains a Pytorch model for single epoch.

    Turns the given pytorch model into training mode and then runs
    through all the steps of training(forward pass, loss, optimizer step).

    Args:
        model: A PyTorch model to use for training.
        dataloader: A Dataloader instance to use for training.
        loss_fn: loss_fn to use for training.
        optimizer: Optimizer to use in training.
        device: A target device to compute on.

    Returns:
        A tuple in form of training_accuracy, training loss.
    """
    model.train()

    # Setup train loss and accuracy values.
    train_loss, train_acc = 0, 0

    # Loop through dataloader data batches.
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device.
        X, y = X.to(device), y.to(device)

        # 1. Forward pass.
        preds = model(X)

        # Calculate and accumulate loss.
        loss = loss_fn(preds, y)
        train_loss += loss.item()

        # optimizer zero grad.
        optimizer.zero_grad()

        # Backpropagation.
        loss.backward()

        # optimizer step.
        optimizer.step()

        train_acc += accuracy_fn(y_true=y, y_pred=preds.argmax(dim=1))

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = "mps"):
    """
    Tests a Pytorch model for single epoch.

    Args:
        model: The model on which testing must be done.
        dataloader: dataloader to use for testing.
        loss_fn: loss function to use while testing.
        device: A target device to compute.

    Returns:
      A tuple in form of test_accuracy, test loss.
    """

    # put model in eval mode.
    model.eval()

    # setup test loss and test accuracy values.
    test_loss, test_acc = 0, 0

    # Turn on inference mode.
    with torch.inference_mode():
        # Loop through dataloader batches.
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device.
            X, y = X.to(device), y.to(device)

            # 1. Forward pass.
            test_preds = model(X)

            # 2. accumulate loss and accuracy.
            loss = loss_fn(test_preds, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_preds.argmax(dim=1))

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc