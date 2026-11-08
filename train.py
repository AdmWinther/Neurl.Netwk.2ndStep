"""Train the sentiment model: adjust its numbers so its predictions match the labels."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import SentimentModel


# Function: train_one_epoch
#   Runs one full pass over the training batches. For each batch the model makes predictions,
#   measures how wrong they are (the loss), and adjusts its numbers to be a little less wrong.
# Parameters:
#   model: the SentimentModel to train.
#   loader: the DataLoader with the training batches.
#   loss_fn: the function that measures how wrong the predictions are.
#   optimizer: the method that adjusts the model's numbers after each batch.
# Output:
#   float: the average loss per review over this pass.
def train_one_epoch(model: SentimentModel, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    # Put the model in training mode.
    model.train()
    total_loss = 0.0

    for ids, labels in loader:
        # Forget the adjustments calculated for the previous batch.
        optimizer.zero_grad()

        # Predict the two scores for every review in this batch, then measure the error.
        scores = model(ids)
        loss = loss_fn(scores, labels)

        # backward() works out in which direction each number should move.
        # step() moves the numbers.
        loss.backward()
        optimizer.step()

        # loss.item() turns the tensor into a plain number. Multiplying by the batch size
        # gives the total loss for this batch, so the last, smaller batch is weighted correctly.
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)
