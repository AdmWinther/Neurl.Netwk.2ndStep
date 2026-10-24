# Prompt feedback log

## 2026-10-24

- **Issue:** The `/init` request asked for build, lint, test, and architecture sections, but the repo contains no code yet, so those sections could not be filled truthfully.
  - **Justification:** Documenting commands that do not exist would invent content.
  - **Status:** New

- **Issue:** The `/init` request writes `CLAUDE.md` directly, which conflicts with the rule "Propose before you write."
  - **Justification:** Asking for a proposal first would have avoided overwriting the existing project brief.
  - **Status:** New
