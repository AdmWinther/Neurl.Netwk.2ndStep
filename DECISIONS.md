# Decisions

Design and scope decisions for the IMDB sentiment classifier exam project
("Application of AI"). Each entry records what was decided and why, so the
reasoning survives between sessions and can be defended at the exam.

Last updated: 2026-10-24

---

## Project

**Scope: binary sentiment classifier for text feedback.**
Trained on the IMDB movie review dataset (50,000 labelled reviews). The
deliverable is a working classifier plus the record of how it was built.

**Constraint: I write no code myself.**
All code is produced by AI, reviewed and approved by me. This constraint is
the actual subject of the project — the course assesses how AI is used, not
how impressive the model is. Sentiment classification was chosen *because*
it is a well-understood problem: nothing about it distracts from the
prompting story.

**Deadlines.** Written description (~5 pages) due end of November 2026.
Exam 14 December 2026: 5-minute presentation, 15 minutes of questions.
New job starts 2 November, so the model is to be finished before that date
and the report written during November.

---

## Technical

**Python, not Java.**
Java was the first instinct, since it is my main language. Rejected: a
from-scratch neural network in Java means hand-building tokenization,
embeddings, backpropagation and an optimizer — several hundred lines where
bugs are silent (the model trains, accuracy is just bad). In PyTorch the
same model is roughly 40 lines. With a new job starting 2 November there is
no schedule room for debugging matrix code I did not write. Reading ~40
lines of Python is also less work than reading ~400 lines of Java.

**Small neural network trained from scratch, not a pretrained transformer.**
An embedding layer with mean pooling and a dense output layer. Expected
accuracy roughly 85-88%. Fine-tuning DistilBERT would score higher but adds
a dependency I would have to explain and cannot inspect. An LSTM costs
10-20 minutes per epoch for little gain.

**CPU training.**
Laptop is a Ryzen 7 PRO 7840U with 32 GB RAM. The Radeon 780M is integrated
with no practical PyTorch acceleration path, so training runs on CPU. The
chosen architecture trains in minutes, so this is not a constraint in
practice.

**Three-way split: 20,000 train / 5,000 validation / 25,000 test.**
Validation is carved out of the 25,000-review training half, not the test
half. Validation drives all tuning decisions (epochs, vocabulary size,
sequence length, embedding dimension, learning rate). The test set is run
once, at the end, and that number is what gets reported. Tuning against the
test set would make the reported accuracy an optimistic estimate of nothing
useful.

**Shuffle before splitting, with a fixed random seed.**
Without shuffling, an ordered dataset can produce a single-label validation
set. Without a fixed seed, the split changes per run and the demo would not
match the report.

**Overfitting is tracked per epoch.**
Training and validation loss are printed each epoch. The point where
validation loss starts rising while training loss keeps falling marks the
start of memorization and determines when to stop training.

**venv for the virtual environment.**
Compared venv, uv, Conda and Poetry. venv is built into Python, so nothing
extra has to be installed or explained at the exam. Dependencies are listed in
`requirements.txt`, the file name that `pip install -r` and IDEs expect. The
environment lives in `.venv/`, which is in `.gitignore`.

---

## Demo

**Command line, no UI.**
A menu offering: train a new model, save a model, load a saved model,
classify an entered sentence.

**The demo loads a saved model; it does not train live.**
Training during a presentation means minutes of silence and an unpredictable
result. Loading is instant and repeatable.

**A recording is made as backup, but the demo is given live.**
A recording cannot answer "try a different sentence".

**No Composite pattern for the menu.**
It was used in an earlier image-classifier project and reusing it here was
tempting. Rejected as over-engineering: the menu architecture is not what is
being assessed, and a three-option CLI does not need a pattern. Logged as a
deliberate scope decision.

**Three known failure cases prepared.**
Negation ("not bad at all"), sarcasm, and mixed reviews are weak spots for a
bag-of-embeddings model. Walking into the Q&A with sentences the model gets
wrong, and an explanation of why, is stronger than claiming it works.

---

## AI workflow

**Claude Code, run locally.**
Claude Code on the web runs in the cloud; the model must be trained and
saved on the laptop used for the demo.

**VS Code as editor.**
Better Python support than IntelliJ IDEA, and the Claude Code extension
gives inline diffs — useful when the whole job is reviewing code written by
someone else.

**Plan mode, always. Never acceptEdits or --dangerously-skip-permissions.**
The approval prompts are the review mechanism. Note: permission mode is a
user-side control (Shift+Tab); an instruction in CLAUDE.md cannot enforce
it, so it has to be checked manually.

**CLAUDE.md written by hand, not generated with /init.**
`/init` generates a starter file by exploring an existing codebase — little
use on an empty project, and the governing instructions are more defensible
at the exam if they are mine.

**Q5 as a skill; prompt feedback in CLAUDE.md.**
Q5 (a five-question interview before major decisions, one question at a
time, no conclusion until all five are answered) is occasional and invoked,
so it lives in `.claude/skills/q5/SKILL.md`. The prompt-feedback rule must
apply to every prompt unconditionally, which a skill cannot guarantee, so it
lives in CLAUDE.md. Matching the mechanism to the kind of instruction is
itself part of the exam story.

**Q5 revised after drafting.**
The first draft forbade commenting on answers. Rejected: a correction
mid-interview (Java to Python) is exactly what made the interview valuable.
The rule now allows comment when an answer is wrong, impossible, or would
make the remaining questions pointless.

**Three-layer record of the process.**
1. Raw transcripts — `/export` at the end of each working session. Exact
   proof. Not reconstructed at the end, because long sessions get compacted
   and early prompts come back paraphrased or missing.
2. `PROMPT_LOG.md` — curated per step: my prompt, the feedback on it, what
   was built, whether it worked, what changed next. This is what gets quoted
   in the report. Spot-checked against the raw exports.
3. Git commits — an unedited timeline of the iterate-and-fix loop.

**`PROMPT_FEEDBACK.md` tracks prompt quality over time.**
Date, issue, and whether it was new, repeated, or improved. Read at session
start. A new session has no memory of earlier prompts, so the comparison
only works if the history is a file. Doubles as evidence of improvement
across five weeks.

---

## Open questions

- Whether `PROMPT_FEEDBACK.md` is committed to the repo or kept out of it.
- What the AI should do when my answer to a Q5 question is "I don't know" —
  offer options, or move on.
