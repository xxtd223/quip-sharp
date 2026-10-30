"""Activation PTQ checks. Set QUIP_ACTIVATION_CUDA_TEST=1 for native GPU tests."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from lib import codebook
from lib.linear.activation_quantization import (
    CODEBOOKS, E8ActivationQuantizer, activation_config,
    configure_activation_quantization)
from lib.linear.quantized_linear import QuantizedLinear
from lib.utils.matmul_had import matmul_hadU_cuda


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(2026)

    def test_disabled_is_identity(self):
        x = torch.randn(3, 17, requires_grad=True)
        self.assertIs(E8ActivationQuantizer()(x), x)

    def test_all_bits_and_no_persistent_tables(self):
        x = torch.randn(136, 3).T
        for bits in (2, 3, 4):
            q = E8ActivationQuantizer(dict(bits=bits)).half()
            y = q(x.half())
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(y.dtype, torch.float16)
            self.assertTrue(torch.isfinite(y).all())
            self.assertGreater((y.float() - x).square().mean().item(), 0)
            self.assertEqual(dict(q.state_dict()), {})
            self.assertEqual(q.get_codebook(x.device).grid_part.dtype, torch.float32)

    def test_zero_and_short_group_rms(self):
        for bits in (2, 3, 4):
            q = E8ActivationQuantizer(dict(bits=bits))
            x = torch.zeros(2, 136)
            torch.testing.assert_close(q(x), x, rtol=0, atol=0)
            x[:, 128:] = 2
            y = q(x)
            cb = q.get_codebook(x.device)
            expected = cb.quantize(torch.full((1, 8), cb.opt_scale), return_idx=False) * (2 / cb.opt_scale)
            torch.testing.assert_close(y[:, :128], torch.zeros(2, 128), rtol=0, atol=0)
            torch.testing.assert_close(y[:, 128:], expected.expand(2, -1))

    def test_tokens_independent_and_scale_equivariant(self):
        x = torch.randn(2, 256)
        q = E8ActivationQuantizer(dict(bits=3))
        y = q(x)
        changed = x.clone()
        changed[1] *= 100
        torch.testing.assert_close(q(changed)[0], y[0], atol=0, rtol=0)
        torch.testing.assert_close(q(x * 4), y * 4)

    def test_chunk_limit_and_per_token_scale(self):
        x = torch.randn(2, 136)
        for bits in (2, 3, 4):
            q = E8ActivationQuantizer(dict(bits=bits, group_size=0, chunk_size=7))
            cb = q.get_codebook(x.device)
            sizes = []
            original = cb.quantize

            def counted(values, *args, **kwargs):
                sizes.append(len(values))
                return original(values, *args, **kwargs)

            with patch.object(cb, 'quantize', side_effect=counted):
                actual = q(x)
            self.assertLessEqual(max(sizes), 7)
            rms = x.square().mean(-1, keepdim=True).sqrt()
            scale = rms / cb.opt_scale
            expected = original((x / scale).reshape(-1, 8), return_idx=False).reshape_as(x) * scale
            torch.testing.assert_close(actual, expected)

    def test_invalid_settings_and_gradient_path(self):
        for config in (dict(bits=1), dict(group_size=9), dict(chunk_size=0),
                       dict(scale_override=0), dict(resid_scale_override=float('nan')),
                       dict(backend='integer')):
            with self.assertRaises(ValueError):
                activation_config(config)
        q = E8ActivationQuantizer(dict(bits=2))
        with self.assertRaises(ValueError):
            q(torch.randn(1, 17))
        with self.assertRaises(RuntimeError):
            q(torch.randn(1, 16, requires_grad=True))
        with self.assertRaises(ValueError):
            QuantizedLinear(128, 128, 8, 4, False, 'torch.int64', 1,
                            rank=0, train_mode=True, activation_quantization=dict(bits=2))


@unittest.skipUnless(os.environ.get('QUIP_ACTIVATION_CUDA_TEST') == '1',
                     'Enable explicitly on an idle GPU with QUIP_ACTIVATION_CUDA_TEST=1')
class NativeCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        torch.backends.cuda.matmul.allow_tf32 = False

    @staticmethod
    def fill_linear(module, cb, name):
        m, n = module.out_features, module.in_features
        values = torch.randn(m * n // 8, 8, device='cuda')
        _, indices = cb.quantize(values)
        module.Qidxs.copy_(cb.maybe_pack_idxs(indices.reshape(m, n // 8)))
        module.codebook_id.fill_(codebook.get_id(name))
        module.SU.copy_(torch.randn(n, device='cuda').sign())
        module.SV.fill_(0.01)

    @torch.inference_mode()
    def test_all_weight_activation_pairs_against_rotated_reference(self):
        for wbits, name in CODEBOOKS.items():
            cb = codebook.get_codebook(name).cuda()
            module = QuantizedLinear(128, 128, cb.codesz, cb.packsz, cb.pack_out,
                str(cb.idx_dtype), cb.version, rank=0).cuda().half()
            self.fill_linear(module, cb, name)
            packed = module.Qidxs.clone()
            weight = cb.by_idxs(packed).reshape(128, 128).float()
            x = torch.randn(3, 128, device='cuda', dtype=torch.float16)
            # Disabled branch must preserve native results exactly.
            baseline = module(x)
            module.activation_quantizer = E8ActivationQuantizer(dict(bits=0))
            torch.testing.assert_close(module(x), baseline, rtol=0, atol=0)
            for abits in (2, 3, 4):
                module.activation_quantizer = E8ActivationQuantizer(dict(bits=abits))
                for rows in (1, 3):
                    source = x[:rows]
                    rotated = matmul_hadU_cuda(source.float() * module.SU,
                                              module.had_left_T, module.K_left) / 32
                    qx = module.activation_quantizer(rotated).half().float()
                    expected = matmul_hadU_cuda(qx @ weight.T,
                        module.had_right, module.K_right) * module.SV * 32
                    actual = module(source).float()
                    relative = ((actual - expected).norm() / expected.norm()).item()
                    self.assertLess(relative, .01, (wbits, abits, rows, relative))

    @torch.inference_mode()
    def test_tiny_llama_save_reload_prefill_decode_and_configuration(self):
        from model.llama import LlamaForCausalLM
        name = CODEBOOKS[2]
        cb = codebook.get_codebook(name).cuda()
        config = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                            max_position_embeddings=128)
        config._attn_implementation = 'sdpa'
        config.quip_params = dict(codesz=cb.codesz, packsz=cb.packsz, pack_out=cb.pack_out,
            idx_dtype=str(cb.idx_dtype), codebook_version=cb.version, codebook=name,
            lora_rank=0, rescale_WH=False)
        model = LlamaForCausalLM(config).cuda().half().eval()
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                self.fill_linear(module, cb, name)
        configure_activation_quantization(model, dict(bits=3, group_size=128, chunk_size=256))
        with tempfile.TemporaryDirectory() as directory:
            # Save before lazy native initialization, as in hfize_llama.py.
            model.save_pretrained(directory)
            stored = json.loads((Path(directory) / 'config.json').read_text())
            self.assertEqual(stored['quip_params']['activation_quantization']['bits'], 3)
            loaded = LlamaForCausalLM.from_pretrained(directory, torch_dtype=torch.float16,
                                                      attn_implementation='sdpa').cuda().half().eval()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
            tokens = torch.randint(0, 256, (1, 12), device='cuda')
            actual = model(tokens, use_cache=False).logits
            torch.testing.assert_close(loaded(tokens, use_cache=False).logits, actual, rtol=0, atol=0)
            prefix = loaded(tokens[:, :-1], use_cache=True)
            decode = loaded(tokens[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(decode.logits[:, -1], actual[:, -1], rtol=.03, atol=.003)
            self.assertTrue(torch.isfinite(decode.logits).all())
            from lib.utils.unsafe_import import model_from_hf_path
            with self.assertRaisesRegex(ValueError, 'CUDA Graph'):
                model_from_hf_path(directory)
            overridden, _ = model_from_hf_path(directory, use_cuda_graph=False,
                device_map='cpu', activation_quantization=dict(bits=4))
            self.assertTrue(all(m.activation_quantizer.config['bits'] == 4
                for m in overridden.modules() if isinstance(m, QuantizedLinear)))
            overridden = overridden.cuda().half().eval()
            self.assertTrue(torch.isfinite(overridden(tokens, use_cache=False).logits).all())
        before = set(loaded.state_dict())
        configure_activation_quantization(loaded, dict(bits=0))
        self.assertEqual(set(loaded.state_dict()), before)
        self.assertTrue(all(not m.activation_quantizer.enabled for m in loaded.modules()
                            if isinstance(m, QuantizedLinear)))


if __name__ == '__main__':
    unittest.main()
