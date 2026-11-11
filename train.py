"""Train the sentiment model: adjust its numbers so its predictions match the labels."""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from encode_data import VOCAB_SIZE, build_vocabulary, encode_split, make_loader
from model import SentimentModel
from split_data import load_train_validation_split

SEED = 42  # fixed seed, so every run gives the same result
BATCH_SIZE = 64  # reviews per batch
EMBEDDING_DIM = 100  # length of each word vector
LEARNING_RATE = 0.001  # how big each adjustment step is (Adam optimizer)
PATIENCE = 3  # stop after this many epochs in a row without a lower validation loss
MAX_EPOCHS = 30  # upper limit, in case the validation loss keeps improving
MODEL_FOLDER = Path("models")


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


# Function: evaluate
#   Measures the model on a set of reviews without changing it.
#   Returns the average loss and the accuracy (the share of reviews predicted correctly).
# Parameters:
#   model: the SentimentModel to measure.
#   loader: the DataLoader with the reviews to measure.
#   loss_fn: the function that measures how wrong the predictions are.
# Output:
#   tuple[float, float]: the average loss per review, and the accuracy.
def evaluate(model: SentimentModel, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    # Put the model in evaluation mode, the counterpart of model.train().
    model.eval()
    total_loss = 0.0
    correct = 0

    # no_grad() tells PyTorch that no adjustments will be made, which saves memory and time.
    with torch.no_grad():
        for ids, labels in loader:
            scores = model(ids)
            total_loss += loss_fn(scores, labels).item() * len(labels)

            # The predicted label is the position of the higher score: 0 (negative) or 1 (positive).
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# Function: train_model
#   Trains the model with early stopping. After each epoch it prints the training loss and the
#   validation loss and accuracy. Training stops when the validation loss has not improved for
#   PATIENCE epochs in a row, or after MAX_EPOCHS. At the end the weights from the epoch with
#   the lowest validation loss are restored.
# Parameters:
#   model: the SentimentModel to train.
#   train_loader: the DataLoader with the training batches.
#   val_loader: the DataLoader with the validation batches.
#   loss_fn: the function that measures how wrong the predictions are.
#   optimizer: the method that adjusts the model's numbers.
# Output: None, the results are printed.
def train_model(model: SentimentModel, train_loader: DataLoader, val_loader: DataLoader,
                loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    # Keep track of the best epoch so far.
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)

        # :.4f means four decimal places, and :.2% means a percentage with two decimals.
        print(f"epoch {epoch}: train loss {train_loss:.4f} | "
              f"validation loss {val_loss:.4f} | validation accuracy {val_accuracy:.2%}")

        # A lower validation loss than before: save this epoch's weights and reset the counter.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == PATIENCE:
                print(f"validation loss has not improved for {PATIENCE} epochs, stopping")
                break

    # Go back to the best epoch's weights, not the weights of the last epoch.
    model.load_state_dict(best_weights)
    print(f"best epoch: {best_epoch} (validation loss {best_val_loss:.4f})")


# Function: save_model
#   Saves the trained model and its vocabulary to disk, so the demo can load them later
#   without training again. The vocabulary is saved as well, because the model only understands
#   the word IDs from the vocabulary it was trained with.
# Parameters:
#   model: the trained SentimentModel.
#   vocab: the vocabulary the model was trained with.
#   folder: the folder to save the two files in.
# Output: None
def save_model(model: SentimentModel, vocab: dict[str, int], folder: Path) -> None:
    # Create the folder if it does not exist yet. parents=True also creates folders above it if needed.
    folder.mkdir(parents=True, exist_ok=True)

    # state_dict() is only the learned numbers of the model, without the code that uses them.
    torch.save(model.state_dict(), folder / "model.pt")

    # The vocabulary is a dictionary from word to ID, which JSON can store as text.
    # encoding="utf-8" makes sure letters outside English are saved correctly.
    with open(folder / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(vocab, file)


# Function: main
#   Runs the whole training: loads the data, builds the vocabulary, trains the model,
#   and saves the model and its vocabulary to the models/ folder.
# Parameters: None
# Output: None
def main() -> None:
    # Fix the random starting numbers of the model, so every run starts from the same place.
    torch.manual_seed(SEED)

    train_split, val_split = load_train_validation_split()
    vocab = build_vocabulary(train_split["text"], VOCAB_SIZE)

    # Encode both splits with the same vocabulary, then cut them into batches.
    # Only the training batches are shuffled.
    train_loader = make_loader(*encode_split(train_split, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(*encode_split(val_split, vocab), batch_size=BATCH_SIZE, shuffle=False)

    # Create the model, the loss function (how wrong the predictions are) and the optimizer (how the numbers are adjusted).
    model = SentimentModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train, printing the losses each epoch, then save the result.
    train_model(model, train_loader, val_loader, loss_fn, optimizer)
    save_model(model, vocab, MODEL_FOLDER)


if __name__ == "__main__":
    main()
