"""Inference-only E8/RVQ activation quantize/dequantize in the rotated basis.

Codeword searches use FP32 and bounded batches. Activations are returned as
floating point; this is an accuracy reference, not a packed activation backend.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


CODEBOOKS = {2: 'E8P12', 3: 'E8P12RVQ3B', 4: 'E8P12RVQ4B'}
DEFAULTS = dict(bits=0, group_size=128, chunk_size=4096,
                scale_override=None, resid_scale_override=None,
                quantizer='e8_rvq', backend='qdq')
# Nonpersistent, shared FP32 search tables: model.half() must not downcast them.
_CODEBOOK_CACHE = {}


def activation_config(config=None):
    config = dict(config or {})
    unknown = config.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown activation quantization settings: {sorted(unknown)}')
    result = {**DEFAULTS, **config}
    if type(result['bits']) is not int or result['bits'] not in (0, 2, 3, 4):
        raise ValueError('activation bits must be 0 (disabled), 2, 3, or 4')
    group = result['group_size']
    if type(group) is not int or group < 0 or group % 8:
        raise ValueError('activation group_size must be 0 (per token) or a positive multiple of 8')
    if type(result['chunk_size']) is not int or result['chunk_size'] <= 0:
        raise ValueError('activation chunk_size must be a positive number of 8D vectors')
    for name in ('scale_override', 'resid_scale_override'):
        value = result[name]
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'activation {name} must be finite and positive')
    if result['quantizer'] != 'e8_rvq' or result['backend'] != 'qdq':
        raise ValueError('Only E8/RVQ activation quantize/dequantize is supported')
    return result


def add_activation_args(parser):
    """None defaults let evaluation preserve settings stored in a model config."""
    parser.add_argument('--activation-bits', '--activation_bits', type=int,
                        choices=[0, 2, 3, 4], default=None,
                        help='Rotated linear-input E8/RVQ QDQ bits; 0 disables it')
    parser.add_argument('--activation-group-size', '--activation_group_size', type=int,
                        default=None, help='RMS scale group in channels (default 128; 0=per token)')
    parser.add_argument('--activation-chunk-size', '--activation_chunk_size', type=int,
                        default=None, help='Maximum 8D vectors per codebook search (default 4096)')
    parser.add_argument('--activation-scale', '--activation_scale', type=float,
                        default=None, help='Normalized activation RMS; default is codebook.opt_scale')
    parser.add_argument('--activation-resid-scale', '--activation_resid_scale', type=float,
                        default=None, help='Override the activation RVQ residual scale')


def activation_overrides(args):
    names = dict(bits='activation_bits', group_size='activation_group_size',
                 chunk_size='activation_chunk_size', scale_override='activation_scale',
                 resid_scale_override='activation_resid_scale')
    return {key: getattr(args, name) for key, name in names.items()
            if getattr(args, name, None) is not None}


class E8ActivationQuantizer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = activation_config(config)

    @property
    def enabled(self):
        return self.config['bits'] != 0

    def get_codebook(self, device):
        key = (device.type, device.index, self.config['bits'])
        if key not in _CODEBOOK_CACHE:
            from lib import codebook
            _CODEBOOK_CACHE[key] = codebook.get_codebook(
                CODEBOOKS[self.config['bits']]).to(device=device, dtype=torch.float32).eval()
        return _CODEBOOK_CACHE[key]

    def forward(self, x):
        if not self.enabled:
            return x
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('E8 activation QDQ is PTQ inference only; gradient fine-tuning is unsupported')
        if not x.is_floating_point() or x.ndim == 0 or x.shape[-1] % 8:
            raise ValueError('E8 activations require floating point input with channels divisible by 8')
        if x.numel() == 0:
            return x
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            cb = self.get_codebook(x.device)
            rho = self.config['scale_override'] or cb.opt_scale
            channels = x.shape[-1]
            group = self.config['group_size'] or channels
            # A final short group uses its actual channel count for RMS.
            padding = (-channels) % group
            values = F.pad(x.float().reshape(-1, channels), (0, padding))
            groups_per_token = values.shape[-1] // group
            values = values.reshape(-1, group)
            output = torch.empty_like(values)
            vectors_per_group = group // 8
            chunk = self.config['chunk_size']
            groups_per_chunk = max(1, chunk // vectors_per_group)
            for start in range(0, values.shape[0], groups_per_chunk):
                stop = min(start + groups_per_chunk, values.shape[0])
                current = values[start:stop]
                if padding:
                    last = torch.arange(start, stop, device=x.device) % groups_per_token == groups_per_token - 1
                    count = torch.where(last, group - padding, group).unsqueeze(-1)
                else:
                    count = group
                rms = (current.square().sum(dim=-1, keepdim=True) / count).sqrt()
                scale = rms / rho
                safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                normalized = (current / safe_scale).reshape(-1, 8)
                reconstructed = torch.empty_like(normalized)
                for offset in range(0, len(normalized), chunk):
                    reconstructed[offset:offset + chunk] = cb.quantize(
                        normalized[offset:offset + chunk], return_idx=False,
                        resid_scale_override=self.config['resid_scale_override'] or -1)
                # Zero RMS maps exactly to zero even if zero is absent from E8P.
                output[start:stop] = reconstructed.reshape_as(current) * scale
            return output.reshape(-1, channels + padding)[:, :channels].reshape(x.shape).to(x.dtype)


def configure_activation_quantization(model, overrides):
    """Configure a loaded QuIP# model before any CUDA graph capture."""
    from .quantized_linear import QuantizedLinear
    if getattr(model, 'built_graph', False):
        raise ValueError('Reset the model CUDA graph before changing activation quantization')
    params = getattr(model.config, 'quip_params', None)
    if params is None:
        raise ValueError('Activation QDQ requires a QuIP# weight-quantized model')
    config = activation_config({**params.get('activation_quantization', {}), **overrides})
    if config['bits'] and hasattr(model, 'graph_device'):
        raise ValueError('E8 activation QDQ requires loading with use_cuda_graph=False')
    modules = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    if config['bits'] and any(m.train_mode for m in modules):
        raise ValueError('Rotated activation QDQ does not support cached train_mode weights')
    if config['bits'] and any(m.in_features % 8 for m in modules):
        raise ValueError('Activation QDQ requires linear input dimensions divisible by 8')
    for module in modules:
        module.activation_quantizer = E8ActivationQuantizer(config)
    params['activation_quantization'] = config
    return config
