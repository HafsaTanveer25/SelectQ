from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from selectq.model_zoo import is_supported_activation_parent


class ActivationStatsCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.layer_names: List[str] = []
        self.current_stats: Dict[str, Tuple[float, float]] = OrderedDict()

    def _hook_fn(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if not torch.is_tensor(output):
                    return
                x = output.detach().float()
                mean = x.mean().item()
                std = x.std(unbiased=False).item()
                self.current_stats[name] = (mean, std)
        return hook

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if is_supported_activation_parent(module):
                self.layer_names.append(name)
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def clear(self) -> None:
        self.current_stats = OrderedDict()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
