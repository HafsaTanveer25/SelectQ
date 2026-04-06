import copy
from pathlib import Path

import torch

from selectq.dataset import build_datasets, build_loader, make_subset
from selectq.evaluate import evaluate
from selectq.hooks import ActivationStatsCollector
from selectq.model_zoo import build_model
from selectq.quantization import collect_activation_ranges, wrap_model_for_ptq
from selectq.selector import learn_selectq_centroids, rank_training_samples, select_topk_indices
from selectq.utils import ensure_dir, get_device, save_json, set_seed


def run_pipeline(cfg) -> None:
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    device = get_device(cfg.quant.eval_on_cpu)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Loading dataset from: {cfg.dataset.root}")
    train_set, val_set = build_datasets(cfg.dataset.root, cfg.dataset.image_size)
    val_loader = build_loader(
        val_set,
        batch_size=cfg.dataset.val_batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    print(f"[Info] Building model: {cfg.model.name}")
    fp_model = build_model(cfg.model.name, pretrained=cfg.model.pretrained).to(device)
    fp_model.eval()

    print("[Stage 0] Evaluating full-precision model")
    fp_metrics = evaluate(fp_model, val_loader, device=device, print_freq=cfg.eval.print_freq)
    print(f"[FP] Top1={fp_metrics['top1']:.3f}, Top5={fp_metrics['top5']:.3f}")

    collector = ActivationStatsCollector(fp_model)
    collector.register()

    print("[Stage 1] Learning SelectQ centroids")
    centroids = learn_selectq_centroids(fp_model, collector, train_set, cfg, device)

    print("[Stage 2] Ranking training samples")
    ranked = rank_training_samples(fp_model, collector, train_set, centroids, cfg, device)
    selected_indices = select_topk_indices(ranked, cfg.selectq.calibration_size)

    indices_path = str(Path(cfg.save_dir) / "selected_indices.json")
    save_json({"selected_indices": selected_indices}, indices_path)
    print(f"[Info] Saved selected indices to: {indices_path}")

    print("[Stage 3] Building calibration subset")
    calib_subset = make_subset(train_set, selected_indices)
    calib_loader = build_loader(
        calib_subset,
        batch_size=cfg.dataset.train_batch_size_stats,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    print("[Stage 4] PTQ with SelectQ-selected calibration data")
    quant_model = copy.deepcopy(fp_model).cpu()
    collector.remove()
    del collector
    torch.cuda.empty_cache()

    quant_model = wrap_model_for_ptq(
        quant_model,
        w_bits=cfg.quant.weight_bits,
        a_bits=cfg.quant.activation_bits,
        asymmetric=cfg.quant.use_asymmetric,
    )
    quant_model = quant_model.to(device)

    quant_model = collect_activation_ranges(
        quant_model,
        calib_loader,
        device=device,
        max_batches=cfg.quant.calibrate_batches,
        asymmetric=cfg.quant.use_asymmetric,
        a_bits=cfg.quant.activation_bits,
    )

    print("[Stage 5] Evaluating quantized model")
    q_metrics = evaluate(quant_model, val_loader, device=device, print_freq=cfg.eval.print_freq)
    print(f"[PTQ+SelectQ] Top1={q_metrics['top1']:.3f}, Top5={q_metrics['top5']:.3f}")

    results = {
        "fp": fp_metrics,
        "ptq_selectq": q_metrics,
        "config_path": cfg.config_path,
        "model": cfg.model.name,
        "w_bits": cfg.quant.weight_bits,
        "a_bits": cfg.quant.activation_bits,
        "calibration_size": cfg.selectq.calibration_size,
        "centroid_count": cfg.selectq.centroid_count,
    }
    results_path = str(Path(cfg.save_dir) / "results.json")
    save_json(results, results_path)
    print(f"[Done] Results saved to: {results_path}")
