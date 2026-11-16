"""Saving, loading and naming of trained models."""

import json
from pathlib import Path

import torch

from model import SentimentModel

MODELS_ROOT = Path("models")


# Function: check_model_name
#   Checks that a name chosen by the user is safe to use as a folder name in models/.
#   Only letters, digits, hyphens and underscores are allowed, so names like "../secret" are refused.
# Parameters:
#   name: the name the user typed.
# Output:
#   bool: True if the name is allowed, False if not.
def check_model_name(name: str) -> bool:
    # An empty name cannot be used as a folder name.
    if name == "":
        return False

    # Check each character. Only letters, digits, "-" and "_" are allowed.
    for character in name:
        if not (character.isascii() and (character.isalnum() or character in "-_")):
            return False

    return True


# Function: save_model
#   Saves the trained model and its vocabulary to models/<name>/, so they can be loaded later.
#   The vocabulary is saved as well, because the model only understands the word IDs from the
#   vocabulary it was trained with.
# Parameters:
#   model: the trained SentimentModel.
#   vocab: the vocabulary the model was trained with.
#   name: the name of the model. It must pass check_model_name.
# Output: None
def save_model(model: SentimentModel, vocab: dict[str, int], name: str) -> None:
    # Refuse names that could point outside the models/ folder.
    if not check_model_name(name):
        raise ValueError(f"invalid model name: {name!r}")

    # Each model gets its own folder. parents=True also creates models/ if it is missing.
    folder = MODELS_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)

    # state_dict() is only the learned numbers of the model, without the code that uses them.
    torch.save(model.state_dict(), folder / "model.pt")

    # The vocabulary is a dictionary from word to ID, which JSON can store as text.
    # encoding="utf-8" makes sure letters outside English are saved correctly.
    with open(folder / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(vocab, file)


# Function: load_model
#   Loads a saved model and its vocabulary from models/<name>/.
# Parameters:
#   name: the name the model was saved under.
# Output:
#   tuple[SentimentModel, dict[str, int]]: the model with its learned numbers, and the vocabulary.
def load_model(name: str) -> tuple[SentimentModel, dict[str, int]]:
    # Refuse names that could point outside the models/ folder.
    if not check_model_name(name):
        raise ValueError(f"invalid model name: {name!r}")

    folder = MODELS_ROOT / name

    # Read the vocabulary first, because its size decides how big the model must be.
    with open(folder / "vocab.json", encoding="utf-8") as file:
        vocab = json.load(file)

    # The embedding size is read from the saved file, so models trained with different sizes still load.
    state = torch.load(folder / "model.pt", map_location="cpu")
    embedding_dim = state["embedding.weight"].shape[1]

    # Create an empty model with the same shape as the saved one, then copy the saved numbers in.
    model = SentimentModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
    model.load_state_dict(state)

    return model, vocab
