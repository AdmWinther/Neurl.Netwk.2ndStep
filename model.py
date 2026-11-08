"""The sentiment model: word vectors, averaged per review, then two scores."""

import torch
from torch import nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output = nn.Linear(embedding_dim, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        word_vectors = self.embedding(ids)
        review_vector = word_vectors.sum(dim=1)
        word_count = (ids != 0).sum(dim=1, keepdim=True).clamp(min=1)
        average = review_vector / word_count
        return self.output(average)
