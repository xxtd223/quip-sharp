"""Check RFFT invariants and reconstruction of packed E8P checkpoints."""

from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from lib import codebook
from lib.algo import quip
from lib.utils.matmul_fft import (matmul_rfft, matmul_rfft_inverse,
                                  random_fft_phase)
from quantize_llama.reproduce_incoherence import restore


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    for n in (8, 128, 4096, 11008, 22016):
        x = torch.randn(3, n, device='cuda', dtype=torch.float64)
        s = random_fft_phase(n, device='cuda', dtype=x.dtype)
        y = matmul_rfft(x, s)
        torch.testing.assert_close(matmul_rfft_inverse(y, s), x,
                                    rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1),
                                    rtol=1e-12, atol=1e-12)
        print('FFT inverse and orthogonality OK:', n, flush=True)
    cb = codebook.get_codebook('E8P12')
    decoder = codebook.get_codebook('E8P12').cuda()
    for mode in ('had', 'fft'):
        torch.manual_seed(0)
        x = torch.randn(128, 128, dtype=torch.float64)
        h = x @ x.T + torch.eye(128, dtype=x.dtype)
        # Explicit symmetry also makes this independent of BLAS reduction order.
        h = ((h + h.T) / 2).float()
        w = torch.randn(128, 128)
        args = SimpleNamespace(
            incoh_mode=mode, use_fp64=False, rescale_WH=False, lora_rank=0,
            full_svd=False, sigma_reg=.01, sigma_reg2=.01, scale_override=.9,
            resid_scale_override=-1, no_use_buffered=False, lowmem_ldlq=False,
            quip_tune_iters=10, save_pfx='/tmp')
        wh, attrs = quip.quantize(h, w, 0, cb, args, 'cuda:0')
        attrs.update(shapes=[w.shape], scales=[1.], fused=False)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'test.pt'
            torch.save(attrs, path)
            restored = restore(path, decoder, args, 'cuda:0')[0]
        relative_error = (restored.float() - wh).norm() / wh.norm()
        assert relative_error < .002, relative_error
        print(mode, 'packed checkpoint restoration relative error:',
              relative_error.item(), flush=True)
        if mode == 'fft':
            hh, ww = h.cuda().double(), w.cuda().double()
            args.use_fp64 = True
            _, hr, wr, su, sv, _ = quip.incoherence_preprocess(hh, ww, args)
            torch.testing.assert_close(hr, hr.T, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(
                quip.incoherence_process(wr, su, sv, None, args), ww,
                rtol=1e-10, atol=1e-10)
            torch.testing.assert_close((wr @ hr * wr).sum(),
                                        (ww @ hh * ww).sum(),
                                        rtol=1e-10, atol=1e-10)
            print('RFFT Hessian symmetry, inverse and proxy preservation OK',
                  flush=True)
    print('ALL TRANSFORM VALIDATIONS PASSED', flush=True)


if __name__ == '__main__':
    main()
