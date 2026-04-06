from typing import List, Sequence, Tuple

from tqdm import tqdm

from selectq.clustering import init_centroids, score_sample, update_centroids
from selectq.dataset import build_loader, make_subset
from selectq.stats import build_random_init_stats, extract_sample_stats_from_batch


def learn_selectq_centroids(
    model,
    collector,
    train_set,
    cfg,
    device,
):
    init_stats = build_random_init_stats(
        model=model,
        collector=collector,
        device=device,
        num_samples=cfg.selectq.random_init_samples,
        image_size=cfg.dataset.image_size,
    )
    centroids = init_centroids(
        collector.layer_names,
        init_stats,
        cfg.selectq.centroid_count,
    )

    update_count = min(cfg.selectq.update_pass_max_samples, len(train_set))
    update_indices = list(range(update_count))
    update_subset = make_subset(train_set, update_indices)
    loader = build_loader(
        update_subset,
        batch_size=cfg.dataset.train_batch_size_stats,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    all_stats = []
    for images, _ in tqdm(loader, desc="[SelectQ] Update pass"):
        images = images.to(device, non_blocking=True)
        batch_stats = extract_sample_stats_from_batch(model, collector, images)
        all_stats.extend(batch_stats)

    centroids = update_centroids(
        centroids=centroids,
        sample_stats_list=all_stats,
        gamma=cfg.selectq.gamma,
        lambda_min=cfg.selectq.lambda_min,
        lambda_max=cfg.selectq.lambda_max,
    )
    return centroids


def rank_training_samples(
    model,
    collector,
    train_set,
    centroids,
    cfg,
    device,
) -> List[Tuple[int, float]]:
    rank_count = min(cfg.selectq.ranking_pass_max_samples, len(train_set))
    ranked = []
    progress = tqdm(range(rank_count), desc="[SelectQ] Ranking pass")
    for idx in progress:
        image, _ = train_set[idx]
        image = image.unsqueeze(0).to(device, non_blocking=True)
        collector.clear()
        _ = model(image)
        sample_stats = dict(collector.current_stats)
        score = score_sample(sample_stats, centroids, cfg.selectq.gamma)
        ranked.append((idx, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def select_topk_indices(ranked: Sequence[Tuple[int, float]], k: int) -> List[int]:
    return [idx for idx, _ in ranked[:k]]
