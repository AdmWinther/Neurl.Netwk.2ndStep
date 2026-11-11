import torch
from pathlib import Path
from evaluate_test import load_model
from encode_data import encode

model, vocab = load_model(Path("models"))
model.eval()
ids = torch.tensor([encode("I minimumasdfsdfsdfwerfaf was ", vocab)])
with torch.no_grad():
    print(torch.softmax(model(ids), dim=1))