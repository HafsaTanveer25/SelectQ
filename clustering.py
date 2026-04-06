from typing import Dict, List, Tuple

import numpy as np

from selectq.utils import cosine_annealed_step

LayerStats = Dict[str, Tuple[float, float]]
CentroidBank = Dict[str, np.ndarray]


def init_centroids(layer_names: List[str], init_stats_list: List[LayerStats], centroid_count: int) -> CentroidBank:
    centroids: CentroidBank = {}
    for layer_name in layer_names:
        pairs = []
        for stats in init_stats_list:
            mean, std = stats[layer_name]
            pairs.append([mean, std])
        pairs = np.asarray(pairs, dtype=np.float32)
        if len(pairs) < centroid_count:
            reps = int(np.ceil(centroid_count / max(len(pairs), 1)))
            pairs = np.tile(pairs, (reps, 1))
        centroids[layer_name] = pairs[:centroid_count].copy()
    return centroids


def knowledge_distance(sample_pair: Tuple[float, float], centroid_pair: np.ndarray, gamma: float) -> float:
    delta_mean = float(sample_pair[0] - centroid_pair[0])
    delta_std = float(sample_pair[1] - centroid_pair[1])
    return delta_mean * delta_mean + gamma * delta_std * delta_std


def nearest_centroid_index(sample_pair: Tuple[float, float], layer_centroids: np.ndarray, gamma: float) -> int:
    distances = [knowledge_distance(sample_pair, c, gamma) for c in layer_centroids]
    return int(np.argmin(distances))


def update_centroids(
    centroids: CentroidBank,
    sample_stats_list: List[LayerStats],
    gamma: float,
    lambda_min: float,
    lambda_max: float,
) -> CentroidBank:
    tmax = max(len(sample_stats_list) - 1, 1)
    for t, sample_stats in enumerate(sample_stats_list):
        step = cosine_annealed_step(
            t=t,
            tmax=tmax,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        for layer_name, pair in sample_stats.items():
            idx = nearest_centroid_index(pair, centroids[layer_name], gamma)
            centroids[layer_name][idx, 0] += step * pair[0]
            centroids[layer_name][idx, 1] += step * pair[1]
    return centroids


def score_sample(sample_stats: LayerStats, centroids: CentroidBank, gamma: float) -> float:
    score = 0.0
    for layer_name, pair in sample_stats.items():
        layer_centroids = centroids[layer_name]
        distances = [knowledge_distance(pair, c, gamma) for c in layer_centroids]
        score += min(distances)
    return float(score)
