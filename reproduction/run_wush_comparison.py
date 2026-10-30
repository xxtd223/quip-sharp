"""Reproducible WUSH/Hadamard GPTQ comparison against the QuIP# WT2 evaluator.

Use the wush conda environment. Calibration and inference both use FP16 and
4096 tokens. Quantization uses the existing WUSH routines without changing
their numerical algorithm. Only one calibration-activation buffer is kept;
the original single-candidate block-loss diagnostic is omitted to save RAM.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
WUSH = ROOT.parent / "WUSH"
OUT = ROOT / "reproduction/wush-comparison"
TEST_DATA = ROOT / "reproduction/data/wikitext-2-raw-test.parquet"
CALIBRATION_DATA = WUSH / "datasets/fineweb-edu"
EVALUATOR = ROOT / "quip-sharp/quantize_llama/reproduce_incoherence.py"
MODELS = {
    "7b": Path("/mnt/model/llama-2-7b-hf"),
    "13b": ROOT / "models/llama-2-13b-hf",
}
EXPECTED_BASELINES = {"7b": 5.116333831331295, "13b": 4.573694685252182}
LAYERS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
          "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
CONTEXT = 4096
SAMPLES = 1024
sys.path.insert(0, str(WUSH))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def reset_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def tokenizer_for(name):
    return AutoTokenizer.from_pretrained(
        MODELS[name], use_fast=False, local_files_only=True)


def prepare():
    """CPU-only data verification; cache exactly the same samples for all jobs."""
    OUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((OUT / "quip-evaluation-data.json").read_text())
    records = {}
    for name in MODELS:
        tokenizer = tokenizer_for(name)
        tokens = tokenizer("\n\n".join(pd.read_parquet(TEST_DATA)["text"].tolist()),
                           return_tensors="pt").input_ids
        record = dict(tokens=tokens.numel(),
                      token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                      parquet_sha256=digest(TEST_DATA))
        if record != reference:
            raise ValueError(f"{name}: evaluation data differs from QuIP#: {record}")
        records[name] = record
    if digest(MODELS["7b"] / "tokenizer.model") != digest(MODELS["13b"] / "tokenizer.model"):
        raise ValueError("Sharing calibration IDs requires identical tokenizer models")
    cache = OUT / "fineweb-edu-1024x4096-seed0.pt"
    metadata_path = OUT / "calibration.json"
    if not cache.exists():
        from src.utils.data_utils import get_data
        reset_seed()
        logging.info("Preparing %s FineWeb-Edu calibration samples, context=%s", SAMPLES, CONTEXT)
        samples = get_data(str(CALIBRATION_DATA), tokenizer_for("7b"), CONTEXT, SAMPLES, 0)
        tokens = torch.cat(samples, dim=0).contiguous()
        if tokens.shape != (SAMPLES, CONTEXT):
            raise ValueError(f"Unexpected calibration shape {tokens.shape}")
        atomic_torch(cache, tokens)
        metadata = dict(
            dataset=str(CALIBRATION_DATA), samples=SAMPLES, context=CONTEXT, seed=0,
            sampler="WUSH get_fineweb_edu; streaming shuffle buffer=1000; random document crop",
            token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            cache_sha256=digest(cache), tokenizer_model_sha256=digest(MODELS["7b"] / "tokenizer.model"),
            source_files=[dict(path=str(p), bytes=p.stat().st_size,
                               mtime_ns=p.stat().st_mtime_ns)
                          for p in sorted(CALIBRATION_DATA.rglob("*.parquet"))],
            created_at=now(),
        )
        atomic_json(metadata_path, metadata)
    else:
        metadata = json.loads(metadata_path.read_text())
        if metadata["cache_sha256"] != digest(cache):
            raise ValueError("Calibration cache checksum mismatch")
    atomic_json(OUT / "data-verification.json", dict(evaluation=records, calibration=metadata))
    logging.info("Data verified: 341469 test tokens; calibration cache %s", cache)


def other_gpu_processes():
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True)
    return sorted(int(line.strip()) for line in output.splitlines()
                  if line.strip().isdigit() and int(line.strip()) != os.getpid())


def wait_gpu(state, save):
    if not other_gpu_processes():
        return
    idle_checks = 0
    while idle_checks < 3:
        others = other_gpu_processes()
        if others:
            idle_checks = 0
            state.update(stage="waiting_for_gpu", other_gpu_processes=others)
            save()
            logging.info("WAIT_GPU other_compute_pids=%s", others)
        else:
            idle_checks += 1
        if idle_checks < 3:
            time.sleep(10)


def load_evaluator():
    # Compile the actual existing function, without importing the QuIP# CUDA
    # extension into WUSH's separate environment. This preserves its scorer,
    # tokenizer, context partition, loss dtype and timing boundary exactly.
    tree = ast.parse(EVALUATOR.read_text(), filename=str(EVALUATOR))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(torch=torch, pd=pd, logging=logging, math=math, hashlib=hashlib)
    exec(compile(module, str(EVALUATOR), "exec"), namespace)
    return namespace["evaluate"]


def sources():
    paths = [EVALUATOR, Path(__file__), WUSH / "src/quantization/wush_gptq.py",
             WUSH / "src/utils/wush_utils.py", WUSH / "src/utils/data_utils.py",
             WUSH / "src/quantization/accumulate_hessian.py",
             WUSH / "inference_lib/src/fp_quant/module/triton/mxfp4.py",
             WUSH / "inference_lib/src/fp_quant/module/triton/nvfp4.py",
             WUSH / "src/utils/common_utils.py", WUSH / "src/utils/model_utils.py"]
    return {str(p): digest(p) for p in paths}


@torch.inference_mode()
def verify_driver():
    """Compare the memory-saving orchestration with the actual WUSH entry point."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.quantization.wush_gptq import wush_quantization_gptq

    global SAMPLES, CONTEXT
    saved_shape = SAMPLES, CONTEXT
    SAMPLES, CONTEXT = 2, 32
    results = []
    scratch = OUT / "driver-verification-checkpoints"
    try:
        reset_seed()
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=4, max_position_embeddings=4096,
                             use_cache=False)
        config._attn_implementation = "sdpa"
        initial = LlamaForCausalLM(config).half().eval()
        calibration = torch.randint(0, 128, (SAMPLES, CONTEXT))
        for fmt in ("mxfp4", "nvfp4"):
            for transform in ("wush", "hadamard"):
                original = copy.deepcopy(initial)
                adapted = copy.deepcopy(initial)
                resumed = copy.deepcopy(initial)
                args = argparse.Namespace(cpu_offload_activations=True,
                                          cpu_offload_modules=True, gptq=True,
                                          transform_class=transform, format=fmt[:-1],
                                          export_quantized_model=False)
                reset_seed()
                wush_quantization_gptq(original, list(calibration.split(1)), args,
                                       torch.device("cuda"))
                checkpoint_dir = scratch / f"{fmt}-{transform}"
                reset_seed()
                quantize_model(adapted, calibration, fmt, transform, checkpoint_dir, {}, lambda: None)
                quantize_model(resumed, calibration, fmt, transform, checkpoint_dir, {}, lambda: None)
                expected = original.state_dict()
                for candidate in (adapted, resumed):
                    actual = candidate.state_dict()
                    assert expected.keys() == actual.keys()
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0,
                                                   msg=f"{fmt}/{transform}: {name}")
                x = calibration[:1].cuda()
                reference_logits = original.cuda()(x, use_cache=False).logits
                for candidate in (adapted, resumed):
                    logits = candidate.cuda()(x, use_cache=False).logits
                    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
                results.append(dict(format=fmt, transform=transform,
                                    exact_weights_transforms_and_logits=True,
                                    packed_checkpoint_resume_identical=True))
                logging.info("DRIVER_VERIFIED %s %s", fmt, transform)
                del original, adapted, resumed, reference_logits, logits, expected, actual, x
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        SAMPLES, CONTEXT = saved_shape
    shutil.rmtree(scratch)
    return dict(created_at=now(), model="random Llama, 2 layers, hidden=64, intermediate=128",
                calibration_samples=2, context=32, comparisons=results)


@torch.no_grad()
def quantize_model(model, calibration, fmt, transform, checkpoint_dir, record, save):
    from src.quantization.wush_gptq import WUSH as HessianHandle, quantize_layers, dequant_xxfp4
    from src.utils.wush_utils import QuantLinear, BlockTransformQuantizer
    from src.utils.model_utils import ForwardInterrupt
    from src.utils.common_utils import to, maybe_first_element

    group = 32 if fmt == "mxfp4" else 16
    blocks = model.model.layers
    hidden = torch.empty(SAMPLES, 1, CONTEXT, model.config.hidden_size, dtype=torch.float16)
    captured_kwargs = []
    capture_index = 0

    def capture(module, args, kwargs):
        nonlocal capture_index
        hidden[capture_index].copy_(args[0].cpu())
        if not captured_kwargs:
            captured_kwargs.append(to(kwargs, device="cpu"))
        capture_index += 1
        raise ForwardInterrupt

    record["stage"] = "capture_calibration_inputs"
    save()
    model.model.embed_tokens.cuda()
    model.model.rotary_emb.cuda()
    hook = blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for sample in calibration:
            try:
                model(sample.unsqueeze(0).cuda(), use_cache=False)
            except ForwardInterrupt:
                pass
    finally:
        hook.remove()
        model.model.embed_tokens.cpu()
        model.model.rotary_emb.cpu()
    assert capture_index == SAMPLES
    gpu_kwargs = to(captured_kwargs[0], device="cuda")
    del captured_kwargs
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for index, block in enumerate(blocks):
        wait_gpu(record, save)
        record.update(stage="quantization", current_block=index, total_blocks=len(blocks))
        save()
        started = time.perf_counter()
        block.cuda()
        checkpoint = checkpoint_dir / f"{index:02d}.pt"
        if checkpoint.exists():
            packed = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for name, item in packed.items():
                scale_dtype = torch.float8_e8m0fnu if fmt == "mxfp4" else torch.float8_e4m3fn
                weight = dequant_xxfp4(
                    item["qweight"].cuda(), item["scales"].cuda().view(scale_dtype),
                    item["global_scale"], dtype=torch.float16)
                block.set_submodule(name, QuantLinear(weight, BlockTransformQuantizer(
                    item["transform"].cuda(), fmt)))
            logging.info("RESUME block %s/%s", index + 1, len(blocks))
        else:
            handles = {name: HessianHandle(block.get_submodule(name)) for name in LAYERS}
            hooks = []
            for name, handle in handles.items():
                def update(module, args, output, handle=handle):
                    handle.update(args[0])
                hooks.append(block.get_submodule(name).register_forward_hook(update))
            try:
                for sample_index, activation in enumerate(hidden):
                    block(activation.cuda(), **gpu_kwargs)
                    if (sample_index + 1) % 128 == 0:
                        logging.info("HESSIAN block=%s samples=%s/%s", index, sample_index + 1, SAMPLES)
            finally:
                for hook in hooks:
                    hook.remove()
            packed = {}
            # Quantize from the unmodified reference block, just as the
            # original routine does before installing its single candidate.
            replacements = {}
            for name in LAYERS:
                t = time.perf_counter()
                layers, results = quantize_layers(
                    block.get_submodule(name).weight, hessian=handles[name].hessian,
                    quant_type=fmt, transform_class=transform, force_rtn=False)
                layer, result = layers[0], results[0]
                matrix = layer.activation_quantizer.transform
                assert matrix.shape[-2:] == (group, group)
                assert torch.isfinite(layer.weight).all() and torch.isfinite(matrix).all()
                packed[name] = dict(
                    qweight=result["qweight"].view(torch.uint8).cpu(),
                    scales=result["scale_quant"].view(torch.uint8).cpu(),
                    global_scale=result["global_scale"], transform=matrix.cpu(),
                )
                replacements[name] = layer
                del handles[name].hessian
                logging.info("QUANTIZED block=%s layer=%s seconds=%.2f", index, name, time.perf_counter() - t)
            for name, layer in replacements.items():
                block.set_submodule(name, layer)
            del handles, replacements, results, result, layers, layer, matrix
            atomic_torch(checkpoint, packed)
        del packed
        torch.cuda.synchronize()
        quantization_seconds = time.perf_counter() - started
        # In-place CPU propagation retains just one 32/40 GiB activation
        # buffer. There is only one candidate, so no block-loss selection.
        for sample_index in range(SAMPLES):
            output = maybe_first_element(block(hidden[sample_index].cuda(), **gpu_kwargs))
            hidden[sample_index].copy_(output.cpu())
        del output
        block.cpu()
        torch.cuda.synchronize()
        item = dict(block=index, seconds=time.perf_counter() - started,
                    hessian_and_quantization_seconds=quantization_seconds,
                    checkpoint_sha256=digest(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)
        previous = {v["block"]: v for v in record.get("blocks", [])}
        if index not in previous:
            record.setdefault("blocks", []).append(item)
        record["quantized_linear_modules"] = (index + 1) * len(LAYERS)
        save()
        logging.info("BLOCK_DONE %s/%s seconds=%.2f", index + 1, len(blocks), item["seconds"])
        gc.collect()
        torch.cuda.empty_cache()
    del hidden, gpu_kwargs
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_run(model, tokenizer, state, save, description):
    evaluate = load_evaluator()
    # Transformers 4.40 used by QuIP# explicitly upcasts Llama logits to
    # FP32 before returning. The WUSH environment's newer implementation
    # omits that cast; reproduce the original scoring precision here.
    if not getattr(model, "_quip_logits_fp32_hook", False):
        model.lm_head.register_forward_hook(lambda module, inputs, output: output.float())
        model._quip_logits_fp32_hook = True
    discarded = state.setdefault("discarded_contended_attempts", [])
    while True:
        wait_gpu(state, save)
        state["stage"] = "evaluation_warmup"
        save()
        model.cuda().eval()
        with torch.inference_mode():
            model(torch.ones(1, 8, device="cuda", dtype=torch.long), use_cache=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state["stage"] = "evaluation"
        save()
        started = time.perf_counter()
        metric = evaluate(model, tokenizer, TEST_DATA, CONTEXT, evaluation=description)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        others = other_gpu_processes()
        if not others:
            break
        discarded.append(dict(seconds=seconds, other_pids=others, ppl=metric))
        save()
    assert metric["blocks"] == 83 and metric["scored_tokens"] == 339885
    assert metric["test_tokens"] == 341469
    assert metric["test_parquet_sha256"] == json.loads((OUT / "quip-evaluation-data.json").read_text())["parquet_sha256"]
    state.update(ppl=metric, ppl_evaluation_seconds=seconds,
                 peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                 other_gpu_processes_after_ppl=others, timing_valid_without_detected_contention=True)
    save()


def write_report(state):
    lines = ["# WUSH / Hadamard + GPTQ，W4A4 对照实验", "",
             "评测：相同 WikiText2 文件及 token IDs，4096 context，batch=1，83 区块，339885 计分 token；"
             "FP16、SDPA、eager、无 cache、seed=0、8 CPU 线程、关闭 TF32。", "",
             "校准：WUSH FineWeb-Edu 1024×4096，同一份缓存 token IDs 用于全部配置；"
             "与 QuIP# 的 RedPajama 6144×4096 Hessian 不同。", "",
             "量化：七个 decoder 线性投影，GPTQ；MXFP4 G=32 / E8M0，NVFP4 G=16 / E4M3；"
             "保留原 WUSH 的 QDQ 质量评测路径，计时不能作为融合 FP4 内核性能。", "",
             f"状态：{state['status']}；阶段：{state.get('stage', '')}；更新时间：{state['updated_at']}。", "",
             "| 模型 | 配置 | 分组 | 状态 | WikiText2 PPL | 测评流程 s |", "| --- | --- | ---: | --- | ---: | ---: |"]
    for name in state["models"]:
        quip = 5.423203 if name == "7b" else 4.813931
        lines.append(f"| Llama-2-{name.upper()} | QuIP# W4A4（既有结果） | 128（激活） | complete | {quip:.6f} | — |")
        for key, record in state.get("runs", {}).items():
            if record["model"] != name:
                continue
            metric = record.get("ppl", {})
            ppl = f"{metric['ppl']:.6f}" if metric else "—"
            seconds = f"{record['ppl_evaluation_seconds']:.2f}" if metric else "—"
            lines.append(f"| Llama-2-{name.upper()} | {record['label']} | {record.get('group', '—')} | {record['status']} | {ppl} | {seconds} |")
    lines += ["", "原始记录：`results.json`；日志：`run.log`。进行中的配置不代表已得到测评结果。", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--transforms", nargs="+", choices=["wush", "hadamard"], default=["wush", "hadamard"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    reset_seed()
    with (OUT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        if args.prepare_only:
            return
        result_path = OUT / "results.json"
        fingerprints = sources()
        state = json.loads(result_path.read_text()) if result_path.exists() else dict(
            created_at=now(), runs={}, source_sha256=fingerprints,
            settings=dict(context=CONTEXT, calibration_samples=SAMPLES, seed=0, dtype="float16",
                          attention="sdpa", use_cache=False, warmup_tokens=8, threads=8,
                          allow_fp16_reduced_precision_reduction=False,
                          allow_bf16_reduced_precision_reduction=False,
                          output_logits_dtype="float32 (match Transformers 4.40 QuIP# evaluator)",
                          weight_bits=4, activation_bits=4, gptq=True, transforms=args.transforms,
                          group_sizes=dict(mxfp4=32, nvfp4=16), evaluator=str(EVALUATOR)),
        )
        if state["source_sha256"] != fingerprints:
            raise ValueError("Source changed since this suite started; use an explicit new output directory")
        if state["settings"]["transforms"] != args.transforms:
            raise ValueError("Transform selection changed since this suite started")
        state.update(status="running", models=args.models, runner_pid=os.getpid())
        state["environment"] = dict(
            python=sys.version, executable=sys.executable,
            packages={name: importlib.metadata.version(name) for name in
                      ("torch", "transformers", "triton", "datasets", "sentencepiece", "protobuf")},
            cuda=torch.version.cuda,
            gpu=subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total",
                                         "--format=csv,noheader"], text=True).strip(),
        )
        state["model_files"] = {
            name: dict(config_sha256=digest(path / "config.json"),
                       tokenizer_model_sha256=digest(path / "tokenizer.model"),
                       weight_files=[dict(path=str(p), bytes=p.stat().st_size,
                                          mtime_ns=p.stat().st_mtime_ns)
                                     for p in sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))])
            for name, path in MODELS.items() if name in args.models
        }
        jobs = [("baseline", None, None)] + [
            (f"{fmt}-{transform}", fmt, transform)
            for fmt in ("mxfp4", "nvfp4") for transform in args.transforms]
        for name in args.models:
            for suffix, fmt, transform in jobs:
                state["runs"].setdefault(f"{name}-{suffix}", dict(
                    model=name, model_path=str(MODELS[name]), status="pending",
                    label="FP16 baseline" if fmt is None else f"{transform.upper()}+GPTQ {fmt.upper()} W4A4",
                    format=fmt, transform=transform,
                    group=(32 if fmt == "mxfp4" else 16) if fmt else None,
                ))

        def save():
            state["updated_at"] = now()
            atomic_json(result_path, state)
            write_report(state)

        save()
        try:
            if "driver_verification" not in state:
                wait_gpu(state, save)
                state["stage"] = "verify_driver_against_original_wush"
                save()
                state["driver_verification"] = verify_driver()
                save()
            calibration = torch.load(OUT / "fineweb-edu-1024x4096-seed0.pt", weights_only=True)
            for name in args.models:
                for suffix, fmt, transform in jobs:
                    key = f"{name}-{suffix}"
                    record = state["runs"][key]
                    if record["status"] == "complete":
                        continue
                    state["current_run"] = key
                    wait_gpu(state, save)
                    reset_seed()
                    record.update(status="running", started_at=now())
                    state["stage"] = "loading_model"
                    save()
                    model = AutoModelForCausalLM.from_pretrained(
                        MODELS[name], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                        local_files_only=True, attn_implementation="sdpa").eval()
                    model.config.use_cache = False
                    model.requires_grad_(False)
                    tokenizer = tokenizer_for(name)
                    checkpoint_dir = OUT / "checkpoints" / key
                    if fmt:
                        # Blocks update the run record; also expose the current
                        # inner stage in the suite status file.
                        def save_quantization():
                            state["stage"] = record.get("stage", "quantization")
                            save()
                        started = time.perf_counter()
                        quantize_model(model, calibration, fmt, transform, checkpoint_dir, record, save_quantization)
                        record["quantization_workflow_seconds_this_attempt"] = time.perf_counter() - started
                    evaluate_run(model, tokenizer, state, save,
                                 "WUSH QuantLinear + FP4 activation QDQ -> HF SDPA" if fmt else "original FP16 weights -> HF SDPA; warmup before evaluate timer")
                    for field in ("ppl", "ppl_evaluation_seconds", "peak_gpu_allocated_gib",
                                  "other_gpu_processes_after_ppl", "timing_valid_without_detected_contention",
                                  "discarded_contended_attempts"):
                        if field in state:
                            record[field] = state.pop(field)
                    if fmt is None:
                        delta = abs(record["ppl"]["ppl"] - EXPECTED_BASELINES[name])
                        record["quip_fp16_baseline_absolute_difference"] = delta
                        if delta > 1e-4:
                            raise ValueError(f"FP16 baseline differs from QuIP# by {delta}; investigate before quantization")
                    record.update(status="complete", finished_at=now())
                    save()
                    logging.info("RESULT %s %s", key, json.dumps(record["ppl"]))
                    del model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    if checkpoint_dir.exists():
                        shutil.rmtree(checkpoint_dir)
                        record["temporary_checkpoints_removed_after_evaluation"] = True
                        save()
            state.update(status="complete", stage="complete", finished_at=now())
            save()
        except BaseException as error:
            state.update(status="failed", error=repr(error), failed_at=now())
            if state.get("current_run") in state["runs"]:
                state["runs"][state["current_run"]].update(status="failed", error=repr(error))
            save()
            raise


if __name__ == "__main__":
    main()
