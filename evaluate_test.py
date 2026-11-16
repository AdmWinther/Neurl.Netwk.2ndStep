"""Evaluate the saved sentiment model on the held-out test reviews. Run once, at the end."""

import sys
from pathlib import Path

from datasets import Dataset, load_from_disk
from torch import nn

from encode_data import encode_split, make_loader
from model_io import load_model
from train import BATCH_SIZE, evaluate

TEST_PATH = Path("data") / "imdb" / "test"


# Function: load_test_split
#   Loads the 25,000 test reviews. This is the only place in the project where the test split is loaded.
# Parameters: None
# Output:
#   Dataset: the test reviews, with a "text" column and a "label" column.
def load_test_split() -> Dataset:
    return load_from_disk(str(TEST_PATH))


# Function: main
#   Runs the final test evaluation once: loads the named saved model, encodes the test reviews
#   with the saved vocabulary, and prints the test loss and accuracy.
#   The model name is the first command-line argument, for example: python evaluate_test.py latest
# Parameters: None
# Output: None
def main() -> None:
    model_name = sys.argv[1]

    # Use the vocabulary saved with the model. Never build a new one from the test reviews.
    model, vocab = load_model(model_name)

    test = load_test_split()
    loader = make_loader(*encode_split(test, vocab), batch_size=BATCH_SIZE, shuffle=False)

    test_loss, test_accuracy = evaluate(model, loader, nn.CrossEntropyLoss())
    print(f"test loss {test_loss:.4f} | test accuracy {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
