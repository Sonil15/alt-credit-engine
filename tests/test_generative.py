import random

from synthetic_data.generate_raw_mock import GLOBAL_SEED, generate_user_profile, sample_latent_and_default


def test_generative_default_rate_reasonable():
    rng = random.Random(GLOBAL_SEED)
    labels = []
    for _ in range(200):
        _, default = sample_latent_and_default(rng)
        labels.append(default)
    rate = sum(labels) / len(labels)
    assert 0.05 < rate < 0.95


def test_profile_has_ground_truth():
    profile = generate_user_profile(random.Random(42))
    assert "_ground_truth" in profile
    assert "default_label" in profile["_ground_truth"]
    assert profile["_ground_truth"]["default_label"] in (0, 1)
    assert "protected_group" in profile["_ground_truth"]


def test_profile_reproducible_with_seed():
    p1 = generate_user_profile(random.Random(99))
    p2 = generate_user_profile(random.Random(99))
    assert p1["_ground_truth"]["default_label"] == p2["_ground_truth"]["default_label"]
