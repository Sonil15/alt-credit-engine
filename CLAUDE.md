# CLAUDE.md — project instructions for Claude Code

This is a hackathon project; the goal is to **win**. The full agent guide lives in
[`AGENTS.md`](AGENTS.md) — read it. Key points repeated here so they aren't missed:

## Hard constraints
- **White-label: never name the sponsoring bank or university** anywhere in the repo.
  Use "the bank" / "a public-sector bank" / "Data Fiduciary".

## Pitch-framing protocol (REQUIRED after significant changes)
After any *significant* change (new feature, model/architecture/decision change, a fixed
credibility risk, or a defensible engineering decision), **add or update the matching
entry in [`PITCH.md`](PITCH.md)** using its Framing Recipe and Maintenance Protocol.
*Significant only* — skip typos, refactors, formatting, and dep bumps. If a change makes
an existing `PITCH.md` entry inaccurate, fix that entry.

## Orientation
- `PITCH.md` — judge-facing framing for every significant feature (the win condition).
- `mlmodel.md` — ML strategy: glass-box **EBM champion**; SHAP is removed from the
  decision/explanation path (the `shap_*` names that remain carry EBM term
  contributions, not Shapley values — don't reintroduce SHAP framing).
