import random

from synthetic_data.generate_raw_mock import (
    DEMOGRAPHIC_QUOTAS,
    GLOBAL_SEED,
    assign_demographics,
    generate_user_profile,
    sample_latent_and_default,
)


def _sample_demographics(theta: float = 0.5) -> dict[str, str]:
    return assign_demographics([theta])[0]


def test_generative_default_rate_reasonable():
    rng = random.Random(GLOBAL_SEED)
    labels = []
    for _ in range(200):
        _, default = sample_latent_and_default(rng)
        labels.append(default)
    rate = sum(labels) / len(labels)
    assert 0.05 < rate < 0.95


def test_profile_has_ground_truth():
    profile = generate_user_profile(0.62, 0, _sample_demographics(), random.Random(42))
    assert "_ground_truth" in profile
    assert "default_label" in profile["_ground_truth"]
    assert profile["_ground_truth"]["default_label"] in (0, 1)
    assert "protected_group" in profile["_ground_truth"]


def test_profile_reproducible_with_seed():
    demographics = _sample_demographics()
    p1 = generate_user_profile(0.4, 1, demographics, random.Random(99))
    p2 = generate_user_profile(0.4, 1, demographics, random.Random(99))
    assert p1["_ground_truth"]["default_label"] == p2["_ground_truth"]["default_label"]


def test_stratified_assignment_balances_theta_across_groups():
    rng = random.Random(GLOBAL_SEED)
    thetas = [rng.betavariate(2.0, 2.0) for _ in range(100)]
    assignments = assign_demographics(thetas)
    overall = sum(thetas) / len(thetas)
    for attribute, quotas in DEMOGRAPHIC_QUOTAS.items():
        for group, quota in quotas.items():
            members = [thetas[i] for i, a in enumerate(assignments) if a[attribute] == group]
            assert len(members) == quota
            if quota >= 10:
                group_mean = sum(members) / len(members)
                # Matched theta spread: group means stay near the population mean,
                # except for tilted attributes like geography.
                if attribute != "geography":
                    assert abs(group_mean - overall) < 0.08
                else:
                    assert abs(group_mean - overall) < 0.18
