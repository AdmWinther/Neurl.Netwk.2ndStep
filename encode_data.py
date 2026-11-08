"""Turn review text into lists of numbers the model can read.

The vocabulary is built from the training reviews only.
"""

import re
from collections import Counter

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from split_data import load_train_validation_split

PAD = "<pad>"  # fills short reviews up to MAX_LENGTH
UNK = "<unk>"  # stands in for words that are not in the vocabulary
VOCAB_SIZE = 20000
MAX_LENGTH = 256

# A token is a word (letters, digits, and an apostrophe, so "don't" stays one token)
# or a single punctuation mark.
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("<br />", " ")
    return TOKEN_PATTERN.findall(text)

#function: build_vocabulary
#   Builds a vocabulary from the training texts. The vocabulary is a dictionary that maps each word to a unique ID. The most common words get the lowest IDs.
#Parameters:
#   train_texts: a list of strings, each string is a review from the training set.
#   size: the number of tokens we want to keep in the vocabulary.
#Output:
#   dict[str, int]: a dictionary that maps each word to a unique ID. The most common words get the lowest IDs.
def build_vocabulary(train_texts: list[str], size: int) -> dict[str, int]:
    # Count how many times each token appears in the training texts.
    # It is not possible to tokenize everything in the training texts.
    # So we only consider the most frequent tokens. size is the number of tokens we want to keep in the vocabulary. 
    counts = Counter()
    for text in train_texts:
        for token in tokenize(text):
            counts[token] += 1

    # After finding the most frequent tokens, we build the vocabulary and assign an ID to each word.
    # The vocabulary is a dictionary that maps each word to a unique ID. The most common words get the lowest IDs.
    words = []
    for word, _ in counts.most_common(size - 2):
        words.append(word)
    
    # The PAD token gets ID 0 and the UNK token gets ID 1. The other words get IDs starting from 2.
    # PAD gets number 0 and UNK gets number 1. The other slots go to the most common words.
    vocab = {PAD: 0, UNK: 1}
    for word in words:
        vocab[word] = len(vocab)
    return vocab


# Function: encode
#   Encodes a text into a list of numbers that the model can read using the dictionary that is made by build_vocabulary.
# Parameters:
#   text: a string, the review text to encode.
#   vocab: a dictionary that maps each word to a unique ID.
#   max_length: an integer, the maximum length of the list of numbers. If the text is longer than max_length, it will be truncated.
# Output:
#   list[int]: a list of numbers that the model can read. The list is of length max_length.
def encode(text: str, vocab: dict[str, int], max_length: int = MAX_LENGTH) -> list[int]:
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize(text)][:max_length]
    ids = []
    # Fill the list with the IDs of the tokens in the text, up to max_length.
    # Break the text to words.
    for token in tokenize(text):
        # If the list is already at max_length, stop adding more words.
        # The text will be truncated to max_length tokens. If it is longer than max_length, the extra words will be ignored.
        if len(ids) == max_length:
            break
        # For each word, look up its ID in the vocabulary. 
        #   If the word is in the dictionary, use its ID. Otherwise
        #   If the word is not in the dictionary, use the ID for UNK.
        ids.append(vocab.get(token, vocab[UNK]))

    # If the list is shorter than max_length, fill the rest of the list with the ID for PAD.
    while len(ids) < max_length:
        ids.append(vocab[PAD])

    return ids


# Function: encode_split
#   Turns a whole split (for example the training reviews) into two PyTorch tensors:
#   one with the word IDs of every review, and one with the label of every review.
# Parameters:
#   split: a Dataset with a "text" column (the review) and a "label" column (0 = negative, 1 = positive).
#   vocab: a dictionary that maps each word to a unique ID.
# Output:
#   tuple[torch.Tensor, torch.Tensor]: the IDs (one row of MAX_LENGTH numbers per review) and the labels (one number per review).
def encode_split(split: Dataset, vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    # Encode each review into a list of IDs. Every list has the same length, MAX_LENGTH.
    all_ids = []
    for text in split["text"]:
        all_ids.append(encode(text, vocab))

    # Convert the Python lists into PyTorch tensors.
    # dtype=torch.long means whole numbers, which the embedding layer requires.
    ids = torch.tensor(all_ids, dtype=torch.long)
    labels = torch.tensor(split["label"], dtype=torch.long)
    return ids, labels


# Function: make_loader
#   Cuts the reviews into batches, so the model processes a few reviews at a time.
#   With shuffle=True the order of the reviews is mixed again every epoch.
# Parameters:
#   ids: a tensor with one row of word IDs per review.
#   labels: a tensor with one label per review.
#   batch_size: how many reviews go into one batch (starting value 64).
#   shuffle: True for training, False for validation.
#   seed: a fixed number for the shuffle, so every run uses the same order.
# Output:
#   DataLoader: an object that gives the batches one at a time.
def make_loader(ids: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    # Pair each row of IDs with its label, so they stay together when batches are cut.
    dataset = TensorDataset(ids, labels)
    # A generator is PyTorch's random number source. Seeding it fixes the shuffle order.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


# Function: main
#   The main function of the script. It loads the training and validation data, builds the vocabulary, and prints some statistics about the data.
#   It also encodes the first review in the training set and prints the first 20 IDs.
# Parameters: None
# Output: None
def main() -> None:
    train, validation = load_train_validation_split()
    vocab = build_vocabulary(train["text"], VOCAB_SIZE)
    print(f"vocabulary size: {len(vocab)}")

    for name, split in [("train", train), ("validation", validation)]:
        tokens = [token for text in split["text"] for token in tokenize(text)]
        unknown = sum(1 for token in tokens if token not in vocab)
        print(f"{name}: {unknown / len(tokens):.2%} of words are unknown")

    print(encode(train[0]["text"], vocab)[:20])


if __name__ == "__main__":
    main()
