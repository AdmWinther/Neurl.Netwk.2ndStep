"""Download the IMDB movie review dataset once and save it to data/imdb.

After this has run, training and the demo can read from data/imdb
without an internet connection.
"""

from pathlib import Path

from datasets import DatasetDict, load_dataset

DATA_DIR = Path("data") / "imdb"


def main() -> None:
    # Downloads from Hugging Face the first time, then reuses its own cache.
    full = load_dataset("stanfordnlp/imdb")

    # Keep only the labelled splits. The 50,000 "unsupervised" reviews are unused.
    dataset = DatasetDict({"train": full["train"], "test": full["test"]})

    # Copy the data into the project folder as plain files.
    DATA_DIR.parent.mkdir(exist_ok=True)
    dataset.save_to_disk(str(DATA_DIR))

    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} reviews")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
