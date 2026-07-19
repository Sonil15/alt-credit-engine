import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN


def latlong_to_pincode(lat: float, long: float, cell_deg: float = 0.5) -> str:
    """Deterministic mock reverse-geocoder: map lat/long to a stable 6-digit PIN."""
    row = int((lat - 8.0) / cell_deg)
    col = int((long - 68.0) / cell_deg)
    code = 110000 + (row * 60 + col) * 7 % 889999
    return str(code)


def historical_spatial_variance(orders: list[dict[str, Any]]) -> dict[str, float]:
    """Compute normalized Shannon entropy of delivery_pin_code frequency distribution."""
    pins = [o.get("delivery_pin_code") for o in orders if o.get("delivery_pin_code")]
    if not pins:
        return {"historical_spatial_variance": 0.0, "distinct_pin_codes": 0.0}

    counts = Counter(pins)
    total = len(pins)
    k = len(counts)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    norm = entropy / math.log2(k) if k > 1 else 0.0

    return {
        "historical_spatial_variance": float(norm),
        "distinct_pin_codes": float(k),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def clean_geo(raw_data: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate shipping address drift from lat/long coordinates.
    
    We map each coordinate to a stable PIN code and calculate:
    - spatial_variance_score: the normalized Shannon entropy of delivery locations.
    - anchor_count: the count of unique shipping locations.
    """
    if not raw_data:
        return {"spatial_variance_score": 0.0, "anchor_count": 0.0}

    pins = []
    for p in raw_data:
        if "lat" in p and "long" in p:
            pins.append(latlong_to_pincode(float(p["lat"]), float(p["long"])))

    if not pins:
        return {"spatial_variance_score": 0.0, "anchor_count": 0.0}

    counts = Counter(pins)
    total = len(pins)
    k = len(counts)

    # Calculate normalized Shannon Entropy
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    norm_entropy = entropy / math.log2(k) if k > 1 else 0.0

    return {
        "spatial_variance_score": float(norm_entropy),
        "anchor_count": float(k),
    }
