import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def clean_geo(raw_data: list[dict[str, Any]]) -> dict[str, float]:
    """Cluster geolocation points and compute spatial variance from anchor centroids."""
    if not raw_data:
        return {"spatial_variance_score": 0.0, "anchor_count": 0.0}

    coords = np.array([[float(p["lat"]), float(p["long"])] for p in raw_data if "lat" in p and "long" in p])
    if coords.size == 0:
        return {"spatial_variance_score": 0.0, "anchor_count": 0.0}

    clustering = DBSCAN(eps=0.01, min_samples=3).fit(coords)
    labels = clustering.labels_

    centroids: list[tuple[float, float]] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_points = coords[labels == label]
        centroid = cluster_points.mean(axis=0)
        centroids.append((float(centroid[0]), float(centroid[1])))

    if not centroids:
        centroid = coords.mean(axis=0)
        centroids = [(float(centroid[0]), float(centroid[1]))]

    distances: list[float] = []
    for lat, lon in coords:
        nearest = min(_haversine_km(lat, lon, c_lat, c_lon) for c_lat, c_lon in centroids)
        distances.append(nearest)

    spatial_variance = float(np.mean(distances)) if distances else 0.0

    return {
        "spatial_variance_score": spatial_variance,
        "anchor_count": float(len(centroids)),
    }
