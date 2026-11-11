# Future improvements

Ideas that were considered but not built in the current version, and things to
try if there is time. Each entry says what the idea is, why it was deferred,
and which decision in DECISIONS.md it relates to.

Use this file for:
- **Exam preparation:** the examiners often ask "what would you do differently?"
  or "what are the limitations?". The entries here are the answers.
- **Learning and extra training:** a to-do list for trying more techniques
  after the main project is working.

When a choice is made between two approaches, the decision goes in
DECISIONS.md. If the rejected approach is worth revisiting, it gets an entry
here, and both entries link to each other.

Entry format:

```
## Title
- **Status:** Open / Done / Dropped
- **Idea:** what would be done
- **Why deferred:** the reason it is not in the current version
- **Related decision:** link to the DECISIONS.md entry
- **What to learn / try:** how to test it, or what to study
- **Exam angle:** a likely question and a short answer
```

---

## Subword tokenization (for example BPE or WordPiece)
- **Status:** Open
- **Idea:** Split words into smaller pieces, such as `unbelievably` into
  `un`, `believ`, `ably`. Unknown words then become known pieces, so the
  model rarely sees an "unknown" token.
- **Why deferred:** The first version uses simple word splitting on spaces and
  punctuation. It is easy to explain, and it needs no extra library. Subword
  tokenization needs a tokenizer to be trained or downloaded, which adds
  complexity.
- **Related decision:** "Word-level tokenization for the first version" in
  DECISIONS.md (entry dated 2026-10-25, "Word-level tokenization for the first version").
- **What to learn / try:** Train a small BPE tokenizer with the `tokenizers`
  library on the training reviews only, then compare validation accuracy
  against the word-level version. Compare how often the unknown token appears.
- **Exam angle:** "Why did you not use subword tokenization?" Answer: the
  model's accuracy is limited by the small network and bag-of-embeddings
  design, not by unknown words. A subword tokenizer would add a dependency and
  complexity without a clear gain for this small model. Verify this with the
  comparison above before using it as an answer.

---

## Pretrained transformer model (for example DistilBERT)
- **Status:** Open
- **Idea:** Fine-tune a model that has already learned language from large
  text collections, instead of training an embedding layer from scratch.
- **Why deferred:** Rejected for the first version. It would likely score
  higher, but it adds a dependency that is hard to inspect or explain, and
  training is slow on a CPU.
- **Related decision:** "Small neural network trained from scratch, not a
  pretrained transformer" in DECISIONS.md.
- **What to learn / try:** Fine-tune DistilBERT on the training split, evaluate
  on the validation split, and compare with the small network.
- **Exam angle:** "Why not a transformer?" Answer: the project's aim is to
  show how AI is used and to explain every part of the model. A pretrained
  model is harder to explain, and the choice is documented in DECISIONS.md.

---

## Save the best validation epoch instead of the last epoch
- **Status:** Open
- **Idea:** After each epoch, `train_model` compares the validation loss with the
  lowest value seen so far. When the new value is lower, it keeps a copy of the
  model's numbers. After the last epoch, it restores that copy, so the saved
  model is the one with the lowest validation loss.
- **How it works today:** `train_model` runs all 10 epochs and does not keep
  track of the best one. The model that `save_model` writes is the one after
  the last epoch (epoch 10). In the first run, the validation loss was lowest at
  epoch 9 (0.2916) and rose slightly at epoch 10 (0.2937). The saved model is
  therefore the one after the slight rise, although the difference is small.
- **How it should work after the update:** `train_model` remembers the epoch
  with the lowest validation loss and prints which epoch that was. At the end
  it restores the weights from that epoch before `save_model` is called. The
  test set is still not involved, because the choice is made only from the
  validation loss.
- **Why deferred:** Chosen against for the first version. Saving the last epoch
  is simpler to explain, and the difference in this run is small (about 0.002
  in loss and 0.06 percentage points in accuracy, which is slightly better
  without the change). The change adds state to the training loop and a copy of
  the weights.
- **Related decision:** the choice to save the last epoch, which is pending an
  entry in DECISIONS.md.
- **What to learn / try:** Search for "early stopping" and "model checkpointing".
  Run training with a patience setting, which stops once the validation loss has
  not improved for a fixed number of epochs, and compare it with the best-epoch
  method.
- **Exam angle:** "How did you decide when to stop training?" Answer: the
  validation loss is printed each epoch. The first version saves the final
  epoch. The improvement would save the epoch with the lowest validation loss,
  so the choice uses only the validation data. Mention the epoch 9 and epoch 10
  numbers from the first run.
