"""Small offline CUDA checks for the local QuIP# environment.

Run with: .venv/bin/python smoke_test.py
No model or dataset downloads are needed.
"""

import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "quip-sharp"))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-flash", action="store_true", help="Check the core QuIP# path without FlashAttention")
    options = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is unavailable"
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for name in ["quiptools_cuda", "fast_hadamard_transform", "lib.algo.quip", "lib.linear", "model.llama", "model.mistral"]:
        importlib.import_module(name)
        print(f"PASS import {name}", flush=True)

    from fast_hadamard_transform import hadamard_transform
    from lib import codebook
    from lib.algo import quip
    from lib.linear.quantized_linear import QuantizedLinear
    from model.llama import LlamaForCausalLM
    from transformers import LlamaConfig

    x = torch.randn(2, 128, device="cuda")
    transformed = hadamard_transform(x, scale=128**-0.5)
    restored = hadamard_transform(transformed, scale=128**-0.5)
    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-4)
    print("PASS Hadamard roundtrip (2 x 128)", flush=True)

    args = SimpleNamespace(
        use_fp64=False, rescale_WH=False, incoh_mode="had", lora_rank=0,
        scale_override=-1, resid_scale_override=-1, no_use_buffered=False,
        lowmem_ldlq=False, quip_tune_iters=1,
    )
    calibration = torch.randn(256, 128)
    hessian = calibration.T @ calibration / 256 + 0.1 * torch.eye(128)
    weights = torch.randn(128, 128) / 128**0.5
    with torch.no_grad():
        for name in ["E8P12", "E8P12RVQ3B", "E8P12RVQ4B"]:
            cb = codebook.get_codebook(name)
            quantized, attrs = quip.quantize(hessian, weights, 0, cb, args, device="cuda")
            assert quantized.shape == weights.shape and torch.isfinite(quantized).all()
            layer = QuantizedLinear(
                128, 128, cb.codesz, cb.packsz, cb.pack_out,
                str(cb.idx_dtype), cb.version, rank=0,
            ).cuda()
            for key in ["Qidxs", "SU", "SV"]:
                getattr(layer, key).copy_(attrs[key])
            layer.codebook_id.fill_(codebook.get_id(name))
            for batch in [1, 3]:
                inputs = torch.randn(batch, 128, device="cuda", dtype=torch.float16)
                actual = layer(inputs).float()
                expected = inputs.float() @ quantized.cuda().T
                error = ((actual - expected).norm() / expected.norm()).item()
                assert actual.shape == (batch, 128) and torch.isfinite(actual).all()
                assert error < 0.01, f"{name}, batch={batch}: relative error {error}"
                print(f"PASS {name} quantization + inference, batch={batch}, relative error={error:.6f}", flush=True)

        cb = codebook.get_codebook("E8P12").cuda()
        config = LlamaConfig(
            vocab_size=128, hidden_size=128, intermediate_size=256,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
            pad_token_id=0, attn_implementation="sdpa",
        )
        config.quip_params = dict(
            codesz=cb.codesz, packsz=cb.packsz, pack_out=cb.pack_out,
            idx_dtype=str(cb.idx_dtype), codebook_version=cb.version,
            lora_rank=0, rescale_WH=False,
        )
        model = LlamaForCausalLM(config).cuda().half().eval()
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                m, n = module.out_features, module.in_features
                _, indices = cb.quantize(torch.randn(m * n // cb.codesz, cb.codesz, device="cuda"))
                module.Qidxs.copy_(cb.maybe_pack_idxs(indices.reshape(m, n // cb.codesz)))
                module.codebook_id.fill_(codebook.get_id("E8P12"))
                module.SU.fill_(1)
                module.SV.fill_(n**-0.5)
        tokens = torch.tensor([[1, 3, 5, 7]], device="cuda")
        output = model(tokens, use_cache=True)
        assert output.logits.shape == (1, 4, 128) and torch.isfinite(output.logits).all()
        output = model(torch.tensor([[9]], device="cuda"), past_key_values=output.past_key_values, use_cache=True)
        assert output.logits.shape == (1, 1, 128) and torch.isfinite(output.logits).all()
        print("PASS tiny quantized Llama: 1 layer, prompt + cached token", flush=True)

    if options.skip_flash:
        print("ALL CORE CHECKS PASSED (FlashAttention skipped)", flush=True)
        return

    from flash_attn import flash_attn_func
    qkv = torch.randn(1, 8, 3, 4, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    q, k, v = qkv.unbind(2)
    actual = flash_attn_func(q, k, v, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.01)
    actual.float().square().mean().backward()
    assert qkv.grad is not None and torch.isfinite(qkv.grad).all()
    torch.cuda.synchronize()
    print("PASS FlashAttention forward + backward", flush=True)
    print(f"ALL CHECKS PASSED; peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
