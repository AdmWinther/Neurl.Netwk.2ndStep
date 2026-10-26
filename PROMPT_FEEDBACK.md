# Prompt feedback log

## 2026-10-24

- **Issue:** The `/init` request asked for build, lint, test, and architecture sections, but the repo contains no code yet, so those sections could not be filled truthfully.
  - **Justification:** Documenting commands that do not exist would invent content.
  - **Status:** New

- **Issue:** The `/init` request writes `CLAUDE.md` directly, which conflicts with the rule "Propose before you write."
  - **Justification:** Asking for a proposal first would have avoided overwriting the existing project brief.
  - **Status:** New

- **Issue:** The request asks to "explore the project and propose the next step" and also says "you now must make the virtual environment ... and make requirements.txt". It is unclear whether those files should be created now or only proposed.
  - **Justification:** CLAUDE.md requires a proposal and approval before any write. Stating "propose only" or "create them" would remove the guess. The remembered state was also partly off: `requirements.txt` already exists, and no `.venv` exists yet.
  - **Status:** New

- **Issue:** "Lets move to the next step" does not say which step. The project plan has several possible next steps, and the choice changes what I build.
  - **Justification:** CLAUDE.md requires a proposal and approval before any write. A named step, or "propose the next step", removes the guess.
  - **Status:** Repeated (same underspecified-request issue as the entry above)
