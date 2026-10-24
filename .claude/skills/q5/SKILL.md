---
name: q5
description: Structured five-question interview before a major design decision. Use when the user invokes /q5, or when Claude is about to make a significant architectural or scope decision and the user has agreed to Q5.
---

# Q5

1. Ask the user whether they agree to use Q5. If no, proceed normally.
2. Ask exactly five questions, one at a time. Never list them up front.
3. Generate each question by taking into account the answer to the previous questions and the poject scop.
4. Do not comment on answers, except when an answer is factually wrong or impossible — then say so plainly and re-ask that question.
5. Do not conclude, decide, or implement anything until question five is answered.
6. After the fifth answer, give the conclusion and recommendation.