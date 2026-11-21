"""The command-line menu for the sentiment classifier."""

from pathlib import Path

import download_data
from model_io import MODELS_ROOT, check_model_name, load_model, save_model
from predict import predict_review
from train import train_new_model, validate_model

DATA_ROOT = Path("data") / "imdb"  # where download_data.py saves the data


# Function: data_exists
#   Checks whether the data has been downloaded, so that training and validation can run.
# Parameters: None
# Output:
#   bool: True if both the train and the test folders exist in data/imdb, False if not.
def data_exists() -> bool:
    return (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "test").is_dir()


# Function: require_model
#   Checks that a model is loaded in memory. If none is loaded, prints a message telling the user what to do.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name (None when empty).
# Output:
#   bool: True if a model is loaded, False if not.
def require_model(state: dict) -> bool:
    # If there is no model, say what the user must do first and refuse.
    if state["model"] is None:
        print("No model is loaded. Train (option 2) or load (option 5) a model first.")
        return False

    return True


# Function: train_option
#   Menu option 2. Trains a new model and puts it in the state, so the other options can use it.
#   Refuses with a message if the data has not been downloaded yet.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name.
# Output: None, the state is updated in place.
def train_option(state: dict) -> None:
    # Training needs the data, so refuse with a clear message if it is missing.
    if not data_exists():
        print("Training is not possible because the data is missing. Choose option 1 (download the data) first.")
        return

    # train_new_model prints the progress of each epoch and returns the trained model and its vocabulary.
    model, vocab = train_new_model()

    # Keep the new model in the state. The name is None because it has not been saved yet.
    state["model"] = model
    state["vocab"] = vocab
    state["name"] = None
    print("Training finished. The model is loaded. Choose option 4 to save it.")


# Function: download_option
#   Menu option 1. Downloads the data from Hugging Face and saves it to data/imdb.
#   If the data is already there, asks before downloading again. Network errors are reported, not raised.
# Parameters: None
# Output: None
def download_option() -> None:
    # The data is already there, so ask before replacing it.
    if data_exists():
        answer = input("The data is already downloaded. Download it again? (y/n): ").strip().lower()
        if answer != "y":
            print("Download skipped. The existing data is kept.")
            return

    # Run the download, and report a clear message if it fails, for example because there is no internet.
    try:
        download_data.main()
    # Intentional broad catch: the download can fail in many ways when there is no internet,
    # and the user should see the same clear message in every case. The error text is printed.
    except Exception as error:  # noqa: BLE001
        print(f"Download failed: {error}. Check the internet connection and try again.")
        return

    print("Download finished. The data is saved in data/imdb.")


# Function: validate_option
#   Menu option 3. Measures the model in memory on the 5,000 validation reviews and prints the result.
#   Refuses with a message if there is no model or no data.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name.
# Output: None, the result is printed.
def validate_option(state: dict) -> None:
    # Validation needs a model in memory first.
    if not require_model(state):
        return

    # Validation also needs the data, which is loaded from data/imdb.
    if not data_exists():
        print("Validation is not possible because the data is missing. Choose option 1 (download the data) first.")
        return

    loss, accuracy = validate_model(state["model"], state["vocab"])
    print(f"validation loss {loss:.4f} | validation accuracy {accuracy:.2%}")


# Function: save_option
#   Menu option 4. Saves the model in memory under a name chosen by the user, in models/<name>/.
#   Refuses if no model is loaded or the name is invalid. Asks before overwriting an existing name.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name.
# Output: None, the result is printed and the state is updated in place.
def save_option(state: dict) -> None:
    # Saving needs a model in memory first.
    if not require_model(state):
        return

    name = input("Name for the model (letters, digits, - and _ only): ").strip()
    if not check_model_name(name):
        print(f"'{name}' is not a valid name. Use only letters, digits, '-' and '_'.")
        return

    # A model with this name already exists, so ask before replacing it.
    folder = MODELS_ROOT / name
    if folder.exists():
        answer = input(f"A model named '{name}' already exists. Overwrite it? (y/n): ").strip().lower()
        if answer != "y":
            print("Save cancelled. Nothing was changed.")
            return

    save_model(state["model"], state["vocab"], name)
    state["name"] = name
    print(f"Model saved as '{name}' in {folder}.")


# Function: list_saved_models
#   Returns the names of the models saved in models/. A folder counts only if it contains model.pt.
# Parameters: None
# Output:
#   list[str]: the names, in alphabetical order. An empty list if nothing is saved.
def list_saved_models() -> list[str]:
    # If the models folder does not exist yet, nothing has been saved.
    if not MODELS_ROOT.is_dir():
        return []

    names = []
    # Look at each item in models/, and keep only the folders that contain a model file.
    for folder in sorted(MODELS_ROOT.iterdir()):
        if folder.is_dir() and (folder / "model.pt").exists():
            names.append(folder.name)

    return names


# Function: load_option
#   Menu option 5. Loads a saved model by name and puts it in the state, so the other options can use it.
#   Shows the saved names first. Refuses a name that is not in the list.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name.
# Output: None, the state is updated in place.
def load_option(state: dict) -> None:
    # Nothing to load if no model has been saved yet.
    names = list_saved_models()
    if not names:
        print("No saved models found. Train (option 2) and save (option 4) a model first.")
        return

    # Show the choices, then ask which one to load.
    print("Saved models: " + ", ".join(names))
    name = input("Name of the model to load: ").strip()

    # Accept only a name from the list, so the user cannot point to any other folder.
    if name not in names:
        print(f"There is no saved model named '{name}'. Choose one of the names listed above.")
        return

    model, vocab = load_model(name)
    state["model"] = model
    state["vocab"] = vocab
    state["name"] = name
    print(f"Model '{name}' is loaded.")


# Function: classify_option
#   Menu option 6. Asks for a review and prints whether it is positive or negative,
#   with the model's confidence. Refuses if no model is loaded or if the review is empty.
# Parameters:
#   state: the dictionary that holds the current model, its vocabulary and its name.
# Output: None, the result is printed.
def classify_option(state: dict) -> None:
    # Classifying needs a model in memory first.
    if not require_model(state):
        return

    text = input("Type a review to classify: ").strip()

    # An empty review would only give an answer based on the model's bias, so refuse it.
    if text == "":
        print("The review is empty. Type some text.")
        return

    label, confidence = predict_review(state["model"], state["vocab"], text)
    print(f"Result: {label} (confidence {confidence:.2%})")


# The menu text. Each number matches one handler.
MENU = """
1. Download the data
2. Train a new model
3. Validate the model
4. Save the model
5. Load a saved model
6. Classify a feedback
0. Exit
"""


# Function: main
#   Shows the menu, reads the user's choice, and runs the matching option.
#   Repeats until the user chooses Exit.
# Parameters: None
# Output: None
def main() -> None:
    # The model in memory starts empty. Options 2 and 5 fill it.
    state = {"model": None, "vocab": None, "name": None}

    while True:
        # Show which model is loaded, so the user knows what classification will use.
        if state["name"] is not None:
            status = f"Loaded model: {state['name']}"
        elif state["model"] is not None:
            status = "Loaded model: trained but not saved"
        else:
            status = "No model loaded"
        print(f"\n{status}{MENU}")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            download_option()
        elif choice == "2":
            train_option(state)
        elif choice == "3":
            validate_option(state)
        elif choice == "4":
            save_option(state)
        elif choice == "5":
            load_option(state)
        elif choice == "6":
            classify_option(state)
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print(f"'{choice}' is not an option. Choose a number from the menu.")


if __name__ == "__main__":
    main()
