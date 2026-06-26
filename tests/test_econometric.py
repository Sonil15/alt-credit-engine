import random
import pytest
from models_econometric.ecm_model import compute_resilience_from_series

def test_increasing_salary_detrended():
    # A salary that increases linearly + strongly mean-reverting (alternating) noise
    random.seed(42)
    e = [0.0]
    for _ in range(40):
        e.append(-0.8 * e[-1] + random.normalvariate(0, 10))
    y = [10000.0 + i*1000.0 + e[i] for i in range(40)]
    
    metrics = compute_resilience_from_series(y)
    
    assert metrics["trend_slope"] > 0
    assert metrics["is_stationary"] == 1.0
    assert metrics["resilience_coefficient"] >= 0.5

def test_decreasing_salary_detrended():
    # A salary that decreases linearly + strongly mean-reverting (alternating) noise
    random.seed(42)
    e = [0.0]
    for _ in range(40):
        e.append(-0.8 * e[-1] + random.normalvariate(0, 10))
    y = [50000.0 - i*1000.0 + e[i] for i in range(40)]
    
    metrics = compute_resilience_from_series(y)
    
    assert metrics["trend_slope"] < 0
    assert metrics["is_stationary"] == 1.0
    assert metrics["resilience_coefficient"] >= 0.5

def test_flat_salary():
    # A flat salary with tiny noise
    random.seed(42)
    series = [10000.0 + random.normalvariate(0, 1.0) for _ in range(40)]
    metrics = compute_resilience_from_series(series)
    
    # Flat series: slope should be very close to 0
    assert abs(metrics["trend_slope"]) < 1e-4
    assert metrics["is_stationary"] == 1.0

def test_short_series_fallback():
    series = [10000.0, 12000.0]
    metrics = compute_resilience_from_series(series)
    
    assert metrics["trend_slope"] == 0.0
    assert metrics["resilience_coefficient"] == 0.5
    assert metrics["is_stationary"] == 0.0
