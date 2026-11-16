"""Tune the model settings on the validation set. The test set is never loaded here."""

import torch
from torch import nn

from encode_data import build_vocabulary, encode_split, make_loader
from model import SentimentModel
from split_data import load_train_validation_split
from train import BATCH_SIZE, SEED, train_model


# Function: run_experiment
#   Trains one model with the given settings, using early stopping, and returns its best
#   validation result. Used by the tuning script only. The test split is never loaded here.
# Parameters:
#   vocab_size: how many words are kept in the vocabulary.
#   max_length: how many word IDs each review is cut or padded to.
#   embedding_dim: the length of each word vector.
#   learning_rate: how big each adjustment step is.
#   seed: the random seed for the model's starting numbers and the shuffle.
# Output:
#   tuple[float, float]: the best validation loss, and the validation accuracy at that epoch.
def run_experiment(vocab_size: int, max_length: int, embedding_dim: int,
                   learning_rate: float, seed: int) -> tuple[float, float]:
    # The seed fixes the starting numbers of the model. Each experiment starts from the same place.
    torch.manual_seed(seed)

    train_split, val_split = load_train_validation_split()
    vocab = build_vocabulary(train_split["text"], vocab_size)

    # Only the training batches are shuffled, and the shuffle uses the same seed.
    train_loader = make_loader(*encode_split(train_split, vocab, max_length),
                               batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = make_loader(*encode_split(val_split, vocab, max_length),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train_model(model, train_loader, val_loader, loss_fn, optimizer)


# Function: sweep_setting
#   Changes one setting over a list of values and keeps all the other settings at their starting values.
#   Prints one line per value and returns all the results.
# Parameters:
#   name: the setting to change: "vocab_size", "max_length", "embedding_dim" or "learning_rate".
#   values: the values to try for that setting.
#   settings: the starting value of every setting, as a dictionary.
# Output:
#   list[tuple]: for each value, a tuple (value, validation loss, validation accuracy).
def sweep_setting(name: str, values: list, settings: dict) -> list[tuple]:
    # Refuse a setting name that is not one of the starting settings, so a typo is caught early.
    if name not in settings:
        raise KeyError(f"unknown setting: {name!r}")

    results = []
    for value in values:
        # Copy the starting settings, then change only the one being tested in this run.
        run_settings = dict(settings)
        run_settings[name] = value

        # Every run uses the same seed, so the only difference between runs is the setting.
        loss, accuracy = run_experiment(seed=SEED, **run_settings)
        print(f"{name} = {value}: validation loss {loss:.4f} | accuracy {accuracy:.2%}")
        results.append((value, loss, accuracy))

    return results


# The starting values, as agreed in the plan. Each sweep changes one of them at a time.
START = {"vocab_size": 20000, "max_length": 256, "embedding_dim": 100, "learning_rate": 0.001}

# The values to try for each setting. The starting value is included in every sweep,
# so each sweep also shows that the baseline result is repeatable.
SWEEPS = {
    "vocab_size": [5000, 10000, 20000, 40000],
    "max_length": [128, 256, 400],
    "embedding_dim": [50, 100, 200],
    "learning_rate": [0.0003, 0.001, 0.003],
}


# Function: main
#   Runs the one-at-a-time sweeps on the validation set and prints the results.
#   The results are recorded in DECISIONS.md by hand, after they are read.
# Parameters: None
# Output: None, the results are printed.
def main() -> None:
    # Go through the settings one at a time. Each sweep changes only its own setting.
    for name, values in SWEEPS.items():
        print(f"\n=== sweep: {name} ===")
        sweep_setting(name, values, START)


if __name__ == "__main__":
    main()
