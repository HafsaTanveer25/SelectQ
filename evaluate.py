from typing import Dict

import torch
from tqdm import tqdm


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        result = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def evaluate(model, loader, device, print_freq: int = 50) -> Dict[str, float]:
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(loader, desc="[Eval]")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            top1_sum += acc1.item() * batch_size / 100.0
            top5_sum += acc5.item() * batch_size / 100.0
            total += batch_size

            if (idx + 1) % print_freq == 0:
                print(
                    f"[Eval] step={idx + 1} "
                    f"running_top1={100.0 * top1_sum / total:.3f} "
                    f"running_top5={100.0 * top5_sum / total:.3f}"
                )

    return {
        "top1": 100.0 * top1_sum / total,
        "top5": 100.0 * top5_sum / total,
    }
