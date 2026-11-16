"""Classify a single review as positive or negative."""

import torch

from encode_data import encode
from model import SentimentModel


# Function: predict_review
#   Classifies one review. Returns the label (positive or negative) and how confident the model is.
# Parameters:
#   model: the trained SentimentModel.
#   vocab: the vocabulary the model was trained with.
#   text: the review to classify, as typed by the user.
# Output:
#   tuple[str, float]: the label "negative" or "positive", and the probability of that label (0 to 1).
def predict_review(model: SentimentModel, vocab: dict[str, int], text: str) -> tuple[str, float]:
    # Encode the text the same way as the training reviews. The extra brackets make a batch of one review.
    ids = torch.tensor([encode(text, vocab)], dtype=torch.long)

    # eval() and no_grad() switch off the training behaviour. We only want the answer, not adjustments.
    model.eval()
    with torch.no_grad():
        scores = model(ids)

    # softmax turns the two scores into probabilities that add up to 1.
    # [0] takes the row for our single review.
    probabilities = torch.softmax(scores, dim=1)[0]

    # The larger probability wins. Position 0 is negative and position 1 is positive.
    labels = ["negative", "positive"]
    best = int(probabilities.argmax())
    return labels[best], float(probabilities[best])
