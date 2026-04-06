from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int


def calc_qparams(x: torch.Tensor, bits: int, asymmetric: bool = True) -> QParams:
    x = x.detach().float()
    if asymmetric:
        qmin = 0
        qmax = 2**bits - 1
        xmin = x.min()
        xmax = x.max()
        if float(xmax - xmin) < 1e-8:
            scale = torch.tensor(1.0, device=x.device)
            zp = torch.tensor(0.0, device=x.device)
        else:
            scale = (xmax - xmin) / float(qmax - qmin)
            zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    else:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        xmax = x.abs().max()
        scale = xmax / float(qmax) if float(xmax) > 1e-8 else torch.tensor(1.0, device=x.device)
        zp = torch.tensor(0.0, device=x.device)
    return QParams(scale=scale, zero_point=zp, qmin=qmin, qmax=qmax)


def fake_quantize_tensor(x: torch.Tensor, qp: QParams) -> torch.Tensor:
    q = torch.round(x / qp.scale + qp.zero_point)
    q = torch.clamp(q, qp.qmin, qp.qmax)
    return (q - qp.zero_point) * qp.scale


class QuantWrapper(nn.Module):
    def __init__(self, module: nn.Module, w_bits: int, a_bits: int, asymmetric: bool = True):
        super().__init__()
        self.module = module
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.asymmetric = asymmetric
        self.act_qparams: Optional[QParams] = None
        self.enable_act_quant = False

        weight = self.module.weight.data
        self.weight_qparams = calc_qparams(weight, bits=self.w_bits, asymmetric=self.asymmetric)
        self.module.weight.data.copy_(fake_quantize_tensor(weight, self.weight_qparams))

        if getattr(self.module, "bias", None) is not None:
            self.module.bias.data.copy_(self.module.bias.data)

    def set_activation_qparams(self, qp: QParams) -> None:
        self.act_qparams = qp
        self.enable_act_quant = True

    def forward(self, x):
        y = self.module(x)
        if self.enable_act_quant and self.act_qparams is not None:
            y = fake_quantize_tensor(y, self.act_qparams)
        return y


def wrap_model_for_ptq(model: nn.Module, w_bits: int, a_bits: int, asymmetric: bool = True) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            setattr(model, name, QuantWrapper(child, w_bits=w_bits, a_bits=a_bits, asymmetric=asymmetric))
        else:
            wrap_model_for_ptq(child, w_bits=w_bits, a_bits=a_bits, asymmetric=asymmetric)
    return model


def collect_activation_ranges(model: nn.Module, loader, device, max_batches: int = -1, asymmetric: bool = True, a_bits: int = 8):
    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return
            out = out.detach()
            cur_min = out.min()
            cur_max = out.max()
            if name not in stats:
                stats[name] = (cur_min, cur_max)
            else:
                old_min, old_max = stats[name]
                stats[name] = (torch.minimum(old_min, cur_min), torch.maximum(old_max, cur_max))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, QuantWrapper):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            _ = model(images)

    for handle in handles:
        handle.remove()

    for name, module in model.named_modules():
        if isinstance(module, QuantWrapper) and name in stats:
            xmin, xmax = stats[name]
            fake = torch.stack([xmin, xmax]).to(device)
            qp = calc_qparams(fake, bits=a_bits, asymmetric=asymmetric)
            module.set_activation_qparams(qp)

    return model
