# AGENTS.md — instructions for coding agents (Cursor, Antigravity, Claude, …)

This is a hackathon project; the goal is to **win**. Read this before changing anything.

## Hard constraints
- **Local only — never push to GitHub.**
- **White-label: never name the sponsoring bank or university** anywhere (code, UI,
  README, comments, commits). Use "the bank" / "a public-sector bank" / "Data Fiduciary".

## Pitch-framing protocol (REQUIRED after significant changes)
After any *significant* change — a new feature, a model/architecture/decision change, a
fixed credibility risk, or a defensible engineering decision — **add or update the
matching entry in [`PITCH.md`](PITCH.md)** so the user has judge-facing framing for it.

- Follow the **Framing Recipe** and **Maintenance Protocol** defined in `PITCH.md`.
- *Significant only* — skip typos, pure refactors, formatting, and dep bumps.
- If your change makes an existing `PITCH.md` entry inaccurate, fix that entry.

## Where to look
- `PITCH.md` — judge-facing framing for every significant feature (the win condition).
- `mlmodel.md` — the ML strategy (glass-box EBM champion; SHAP is removed from the
  decision/explanation path — don't reintroduce SHAP framing).
- `README.md` — project overview and run instructions.
