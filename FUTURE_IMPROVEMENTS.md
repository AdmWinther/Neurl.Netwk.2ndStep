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
