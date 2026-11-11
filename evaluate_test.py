"""Evaluate the saved sentiment model on the held-out test reviews. Run once, at the end."""

import json
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from torch import nn

from encode_data import encode_split, make_loader
from model import SentimentModel
from train import BATCH_SIZE, EMBEDDING_DIM, MODEL_FOLDER, evaluate

TEST_PATH = Path("data") / "imdb" / "test"


# Function: load_model
#   Loads the trained model and its vocabulary from the folder that save_model wrote to.
# Parameters:
#   folder: the folder that contains model.pt and vocab.json.
# Output:
#   tuple[SentimentModel, dict[str, int]]: the model with its learned numbers, and the vocabulary.
def load_model(folder: Path) -> tuple[SentimentModel, dict[str, int]]:
    # Read the vocabulary first, because its size decides how big the model must be.
    with open(folder / "vocab.json", encoding="utf-8") as file:
        vocab = json.load(file)

    # Create an empty model with the same shape as the saved one.
    model = SentimentModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM)

    # Copy the saved numbers into the model.
    # map_location="cpu" lets the file load on any computer, even one without a GPU.
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))

    return model, vocab


# Function: load_test_split
#   Loads the 25,000 test reviews. This is the only place in the project where the test split is loaded.
# Parameters: None
# Output:
#   Dataset: the test reviews, with a "text" column and a "label" column.
def load_test_split() -> Dataset:
    return load_from_disk(str(TEST_PATH))


# Function: main
#   Runs the final test evaluation once: loads the saved model, encodes the test reviews
#   with the saved vocabulary, and prints the test loss and accuracy.
# Parameters: None
# Output: None
def main() -> None:
    # Use the vocabulary saved with the model. Never build a new one from the test reviews.
    model, vocab = load_model(MODEL_FOLDER)

    test = load_test_split()
    loader = make_loader(*encode_split(test, vocab), batch_size=BATCH_SIZE, shuffle=False)

    test_loss, test_accuracy = evaluate(model, loader, nn.CrossEntropyLoss())
    print(f"test loss {test_loss:.4f} | test accuracy {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
