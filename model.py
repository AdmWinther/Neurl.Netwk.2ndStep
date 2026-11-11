"""The sentiment model: word vectors, averaged per review, then two scores."""

import torch
from torch import nn

# The SentimentModel class is a PyTorch neural network module that implements a simple sentiment analysis model
# . It takes as input a batch of word IDs (representing reviews) and outputs a batch of sentiment scores (one for each review).
class SentimentModel(nn.Module):

    # The __init__ method initializes the model with an embedding layer and a linear output layer.
    # Parameters:
    #   vocab_size: an integer, the size of the vocabulary (the number of unique words).
    #   embedding_dim: an integer, the size of the word vectors (default is 100). TODO: make this a parameter in the config file.
    # Output: None, this is the constructor of the class, It makes the class ready to be used.
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output = nn.Linear(embedding_dim, 2)

    # The forward method defines how the model processes input data.
    # Parameters:
    #   ids: a tensor of shape (batch_size, sequence_length), containing word IDs for each review.
    # Output: a tensor of shape (batch_size, 2), containing the sentiment scores for each review.
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        word_vectors = self.embedding(ids)
        review_vector = word_vectors.sum(dim=1)
        word_count = (ids != 0).sum(dim=1, keepdim=True).clamp(min=1)
        average = review_vector / word_count
        return self.output(average)
