"""Split the 25,000 IMDB training reviews into 20,000 for training and 5,000 for validation.

Only data/imdb/train is read here. The test split is never loaded, so training
code that uses this file cannot touch the held-out test reviews.
"""

from pathlib import Path

from datasets import Dataset, load_from_disk

TRAIN_PATH = Path("data") / "imdb" / "train"
VALIDATION_SIZE = 5000
# Fixed seed: the same shuffle, and therefore the same split, on every run.
SEED = 42


def load_train_validation_split() -> tuple[Dataset, Dataset]:
    train_reviews = load_from_disk(str(TRAIN_PATH))

    split = train_reviews.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED,
        stratify_by_column="label",
    )

    # train_test_split always names its two parts "train" and "test".
    # Here the second part is the validation set.
    return split["train"], split["test"]


def main() -> None:
    train, validation = load_train_validation_split()

    print(f"train: {len(train)} reviews")
    print(f"validation: {len(validation)} reviews")


if __name__ == "__main__":
    main()
