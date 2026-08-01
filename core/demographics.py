"""Shared code maps for the self-declared demographic attributes used only by the
fairness monitor (never as model inputs - see convergence/fairness.py).

Single source of truth for both the synthetic-seed bootstrap and the live
registration flow, so the two never drift apart.
"""

PROTECTED_GROUP_CODES = {"general": 0, "obc": 1, "sc": 2, "st": 3, "minority": 4}
GENDER_CODES = {"male": 0, "female": 1, "other": 2}
GEOGRAPHY_CODES = {"rural": 0, "semi_urban": 1, "urban": 2}
