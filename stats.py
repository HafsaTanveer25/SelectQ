from typing import Dict, List, Tuple

import torch


def extract_sample_stats_from_batch(
    model,
    collector,
    images: torch.Tensor,
) -> List[Dict[str, Tuple[float, float]]]:
    """
    Simple and faithful debugging-friendly approach:
    run one sample at a time to obtain exact per-sample layer statistics.
    """
    batch_stats = []
    with torch.no_grad():
        for i in range(images.shape[0]):
            collector.clear()
            _ = model(images[i : i + 1])
            batch_stats.append(dict(collector.current_stats))
    return batch_stats


def build_random_init_stats(
    model,
    collector,
    device,
    num_samples: int,
    image_size: int,
) -> List[Dict[str, Tuple[float, float]]]:
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.rand(1, 3, image_size, image_size, device=device)
            collector.clear()
            _ = model(x)
            results.append(dict(collector.current_stats))
    return results
