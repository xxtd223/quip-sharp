"""Real orthogonal randomized FFT from QuIP# Appendix A.2.

Adjacent real coordinates form one complex number. This is a complex FFT on
n/2 coordinates, not torch.fft.rfft (which has different dimensions).
Phases are stored as interleaved real/imaginary coordinates for checkpoints.
"""

import math

import torch


def random_fft_phase(n, *, device, dtype):
    if n % 2:
        raise ValueError('The real representation of the FFT needs an even size')
    theta = torch.rand(n // 2, device=device, dtype=dtype) * (2 * math.pi)
    return torch.stack((theta.cos(), theta.sin()), dim=-1).flatten()


def _complex_pairs(x):
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())


def matmul_rfft(x, phase):
    """Apply T = F diag(phase) to each row, represented over the reals."""
    phase = _complex_pairs(phase.to(device=x.device, dtype=x.dtype))
    y = torch.fft.fft(_complex_pairs(x) * phase, dim=-1, norm='ortho')
    return torch.view_as_real(y).flatten(-2)


def matmul_rfft_inverse(x, phase):
    """Apply T.T to rows; a scalar fused into phase scales the result."""
    phase = _complex_pairs(phase.to(device=x.device, dtype=x.dtype))
    y = torch.fft.ifft(_complex_pairs(x), dim=-1, norm='ortho') * phase.conj()
    return torch.view_as_real(y).flatten(-2)
