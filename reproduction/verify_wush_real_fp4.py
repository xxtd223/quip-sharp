"""Check a local WUSH export; optionally use an unfused, process-local repair.

Run with the WUSH environment. No checkpoint or installed source is modified.
The repair supports inference-only WUSH MXFP4/NVFP4 AbsMax exports; it keeps
packed FP4 weights and QuTLASS GEMM, but runs the block transform separately.
"""

import argparse
import json
from pathlib import Path

import torch
import qutlass
from safetensors import safe_open


def transform_input(x, packed):
    group_size, channels = packed.shape
    if channels != x.shape[-1] or channels % group_size:
        raise ValueError("WUSH transform and input shapes disagree")
    transform = packed.T.contiguous().reshape(-1, group_size, group_size)
    return torch.einsum(
        "...ng,nhg->...nh", x.reshape(*x.shape[:-1], -1, group_size), transform
    ).reshape_as(x).contiguous()


def install_unfused_repair():
    # Install before importing fp_quant: its single import guard otherwise
    # disables all QuTLASS operations when the WUSH symbol is absent.
    if any(name.startswith("fp_quant") for name in __import__("sys").modules):
        raise RuntimeError("Install the repair before importing fp_quant")
    original_nv = qutlass.fusedQuantizeNv

    def mx_wush(x, packed):
        if packed.shape[0] != 32:
            raise ValueError("MXFP4 WUSH requires groups of 32")
        transformed = transform_input(x, packed)
        identity = torch.eye(32, dtype=x.dtype, device=x.device)
        return qutlass.fusedQuantizeMx(transformed, identity, method="abs_max")

    def nv_wush(x, packed, global_scale, *, method="abs_max"):
        if packed.shape[0] == packed.shape[1]:
            return original_nv(x, packed, global_scale, method=method)
        if packed.shape[0] != 16 or method != "abs_max":
            raise ValueError("This repair supports NVFP4 WUSH AbsMax groups of 16")
        transformed = transform_input(x, packed)
        identity = torch.eye(16, dtype=x.dtype, device=x.device)
        return original_nv(transformed, identity, global_scale, method=method)

    qutlass.fusedQuantizeWushMx = mx_wush
    qutlass.fusedQuantizeNv = nv_wush


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--unfused-wush", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config_dict = json.loads((checkpoint / "config.json").read_text())
    quant_config = config_dict["quantization_config"]
    if (quant_config["transform_init"] != "wush"
            or quant_config["forward_method"] != "abs_max"
            or quant_config["pseudoquantization"]
            or quant_config["store_master_weights"]
            or quant_config["backward_dtype"] != "bf16"
            or quant_config["forward_dtype"] not in ("mxfp4", "nvfp4")):
        raise ValueError("Expected an inference-only real FP4 WUSH AbsMax export")

    groups = set()
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("forward_hadamard_matrix"):
                    groups.add(handle.get_slice(key).get_shape()[0])
    if len(groups) != 1:
        raise ValueError(f"Expected a consistent exported group size, found {groups}")
    actual_group = groups.pop()
    print(json.dumps(dict(
        checkpoint=str(checkpoint), gpu=torch.cuda.get_device_name(),
        original_wush_api=hasattr(qutlass, "fusedQuantizeWushMx"),
        configured_group=quant_config["hadamard_group_size"],
        actual_group=actual_group, unfused_wush=args.unfused_wush,
    )), flush=True)
    if args.unfused_wush:
        install_unfused_repair()
        # Correct metadata in memory only; original export stays untouched.
        quant_config["hadamard_group_size"] = actual_group

    from transformers import AutoConfig, AutoModelForCausalLM
    from fp_quant import FPQuantLinear

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.quantization_config = quant_config
    config._attn_implementation = "sdpa"
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, local_files_only=True,
        device_map="cuda", torch_dtype=torch.bfloat16, output_loading_info=True,
    )
    model.eval()
    modules = [m for m in model.modules() if isinstance(m, FPQuantLinear)]
    assert modules and all(m.qweight.dtype == torch.uint8 for m in modules)
    assert all(m.weight is None and m.dqweight is None for m in modules)
    # The local Transformers integration reports obsolete master-weight keys
    # as missing even though qweight has loaded and removed those parameters.
    expected_missing = {
        name + ".weight" for name, m in model.named_modules()
        if isinstance(m, FPQuantLinear)
    }
    assert set(loading_info["missing_keys"]) <= expected_missing, loading_info
    assert not any(loading_info.get(key) for key in
                   ("unexpected_keys", "mismatched_keys", "error_msgs")), loading_info
    with torch.inference_mode():
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        prefill = model(input_ids=tokens, use_cache=True)
        decode = model(
            input_ids=prefill.logits[:, -1:].argmax(-1),
            past_key_values=prefill.past_key_values, use_cache=True,
        )
    torch.cuda.synchronize()
    assert torch.isfinite(prefill.logits).all() and torch.isfinite(decode.logits).all()
    print(json.dumps(dict(
        status="ok", fp4_modules=len(modules),
        prefill_logits=list(prefill.logits.shape),
        decode_logits=list(decode.logits.shape),
        obsolete_master_weight_warnings=len(loading_info["missing_keys"]),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    )), flush=True)


if __name__ == "__main__":
    main()
