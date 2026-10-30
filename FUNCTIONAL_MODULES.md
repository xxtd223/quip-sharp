# 项目功能模块情况

更新日期：2026-10-31。以下以当前源码、已有本机验证记录为准，区分原项目主路径、新增功能、遗留或未接通的实现。“有代码”不等于已经完成全模型验证。本文专门记录项目各功能模块的实现状态、支持边界和验证情况，随代码推进同步维护。开发规格、阶段计划和实验报告保留在本地，不纳入 Git；本文引用的这些材料不随仓库分发。

## 1. 功能概览

| 功能 | 当前状态 | 主要入口 |
| --- | --- | --- |
| Weight-only PTQ | 支持；主要量化 decoder 线性层权重 | `lib/algo/quip.py`、`lib/linear/` |
| Weight-activation PTQ（本地新增） | 可选 E8/RVQ A2/A3/A4 量化后反量化；与 W2/W3/W4 独立组合，默认关闭 | `lib/linear/activation_quantization.py`、三种码本线性前向 |
| E8P² 索引点积 | 尚未实现；当前激活后端仅支持 `qdq`，没有直接从双侧索引计算点积的后端 | `lib/linear/activation_quantization.py` |
| KV cache 量化 | 未实现；有浮点 KV cache 推理 | `model/llama.py` |
| 2 / 3 / 4 bit 权重 | 均有注册码本、索引打包/解码和线性层实现 | `lib/codebook/__init__.py` |
| 旋转/非相干化 | Hadamard 主路径；Fourier 为本地新增；Kronecker 为有缺陷的遗留分支 | `lib/algo/quip.py`、`lib/utils/matmul_*.py` |
| Hessian 感知误差补偿 | 初次 BlockLDLQ 量化 + 可选逐块局部搜索 | `lib/algo/quip.py` |
| 微调恢复精度 | 层内 MSE 重建、端到端教师 logits 蒸馏 | `lib/algo/finetune.py` |
| 低秩误差修正 | 算法函数和参数仍保留；当前量化线性层要求 rank=0，完整主流程不支持启用 | `lib/algo/quip.py`、`lib/linear/quantized_linear.py` |
| 模型适配 | Llama 是完整主路径；Mistral 有遗留模型文件，当前未接通 | `quantize_llama/`、`model/` |
| 测评 | WikiText2/C4 PPL、lm-eval 任务、前向速度、交互生成；另有本地分阶段 profiling | `eval/`、`quantize_llama/profile_incoherence.py` |

## 2. 项目结构与处理流程

| 模块 | 职责 |
| --- | --- |
| `quantize_llama/hessian_offline_llama.py` | 采样校准数据，挂输入 hook，累计激活二阶统计并保存 Hessian；支持多 GPU 分工和可选激活保存 |
| `lib/utils/data_utils.py` | RedPajama/RefinedWeb 采样、Hessian hook、检查点解包等；hook 默认 FP64 累加 |
| `lib/algo/quip.py` | 权重/Hessian 预处理、旋转、BlockLDLQ、逐块局部搜索、打包、误差诊断；保留低秩算法 |
| `lib/codebook/` | E8 格点码字选择、RVQ、索引编码/解码及对应量化线性计算 |
| `lib/linear/` | 普通及融合量化线性层；融合 Q/K/V 和 up/gate；新增旋转后激活 E8/RVQ QDQ、动态 RMS 尺度与配置工具 |
| `lib/algo/finetune.py`、`lib/utils/finetune.py` | 层级重建及端到端微调、优化器、验证集选择和提前停止 |
| `quantize_llama/quantize_finetune_llama.py` | 逐层量化/微调主入口；`--ft_epochs 0` 关闭梯度微调，逐块局部搜索仍由独立参数控制 |
| `quantize_llama/hfize_llama.py` | 将逐矩阵检查点装入项目 Llama 模型，导出 HF 格式并验证加载/生成 |
| `quantize_llama/finetune_e2e_llama.py` | 加载量化模型，构造教师 logits，执行端到端微调；支持模型分片 |
| `model/llama.py` | 项目量化 Llama、GQA、RoPE、Attention、浮点 KV cache；包含 eager/SDPA/FlashAttention2 类 |
| `quiptools/` | CUDA 解码、融合 decode-matvec、E8 残差查表等内核与低层测试 |
| `lib/utils/graph_wrapper.py` | 固定输入形式的 CUDA Graph 捕获/重放；不能据此假定任意形状或动态生成都能直接重放 |

标准流程为：校准并生成 Hessian → 逐矩阵旋转及 Hessian 感知量化 → 可选层内微调 → HF 导出 → 可选端到端微调及再次导出 → 测评/生成。也可以下载作者预计算 Hessian，省略统计生成。校准使用激活不意味着对激活做低比特量化。

## 3. 旋转策略

| 方案 | 算法与源码 | 可用范围与限制 |
| --- | --- | --- |
| 随机 Hadamard / RHT，`had` | 随机符号 SU/SV，配合归一化 Hadamard 旋转权重和 Hessian；`RHT_H`、`RHT_W`，`lib/utils/matmul_had.py` | 原项目默认方案，主 CLI、2/3/4 bit 量化线性层均沿此路径；在线有 CUDA Hadamard。部分非二次幂尺寸由固定小 Hadamard 因子组合支持，并非任意尺寸 |
| 随机 Kronecker/正交 butterfly，`kron` | `rand_ortho_butterfly_noblock` 生成带置换的随机正交变换；预处理显式形成矩阵乘法，逆变换及 `load_quip` 有分支 | 原 CLI 接受，但 `lib/utils/matmul_kron.py` 缺少 `import torch`，最小函数调用已复核触发 `NameError`。标准压缩线性层只执行 Hadamard，修复导入后也仍需适配推理/微调，不能视为可直接运行的完整方案 |
| 随机 Fourier / RFFT，`fft` | 当前工作区新增 `lib/utils/matmul_fft.py`：相邻实数配对成复数、随机复相位、正交归一化 FFT/IFFT；已接入 `quip.py` 正逆预处理 | `reproduce_incoherence.py` 支持 no-FT、2 bit 完整模型量化及还原 FP16 后 PPL；`profile_incoherence.py` 有压缩权重在线适配。原 `quantize_finetune_llama.py` 的 CLI 仍只接受 `had/kron`，`load_quip` 无 FFT 分支，标准 HF 导出/推理也没有通用 FFT 切换，不能直接用于任意码本或微调流程 |

这里 RFFT 是实数表示下的复数 FFT，不是 `torch.fft.rfft` 的实到半谱变换。模型自带的 RoPE 用于位置编码，不是上表的权重量化非相干化方案。未发现独立的可学习正交旋转优化器；微调 SU/SV 是另一个功能，不能等同于训练任意正交旋转矩阵。

## 4. Weight-only 与 weight-activation PTQ

默认仍为 W2/W3/W4 的 weight-only PTQ。新增可选的 E8/RVQ 激活量化后反量化（QDQ）参考实现：`activation_bits=0` 关闭；2/3/4 分别启用 A2/A3/A4，与权重位宽独立选择。A2 使用 E8P12，A3 使用 E8P12RVQ3B（2+1），A4 使用 E8P12RVQ4B（2+2）。这里的 A 位宽是码字编码精度；计算前已重建为浮点，尚无激活打包存储或低比特整数计算内核，不能据此声称激活显存减少或推理加速。

覆盖范围为 Hadamard 主路径的 decoder 线性层输入：融合 Q/K/V、O、融合 up/gate、down。Prefill 与 decode 均执行。Embedding、norm、lm-head、residual、Attention 内部 QKᵀ/AV、softmax 和 KV cache 保持浮点；不是对整个模型所有算子的低比特化。Fourier 实验路径尚未启用这项功能。

### 激活量化位置与尺度

```text
线性层输入 → 可选 scaleWH 重缩放 → SU → Hadamard / 32
           → 可选 E8/RVQ 激活 QDQ → 原 E8 权重解码/浮点矩乘
           → 输出 Hadamard → SV 与输出缩放
```

每 token 沿通道分组计算动态 RMS，默认 128 通道一组；`group_size=0` 为整个 token。尺度组与 8 维码字组不同：同一尺度组内，每 8 个值选择一个码字。设旋转后的输入为 z，`s=RMS(z)/rho`，重建为 `s * Q(z/s)`；全零尺度组返回精确零。通道数必须是 8 的倍数，最后不足一个尺度组时按实际通道数计算 RMS。原权重的 `/32` 缩放位置保持不变；QDQ 的 RMS 归一化在该位置执行。

初始 `rho` 沿用码本的 Gaussian 参数 1.03/0.98/1.03，A3/A4 残差尺度默认 2.04/3.45；这些是实现基线，**未证明对真实激活最优**，W2A2/W2A3/W3A3 已完成完整 PPL。激活参数独立于权重参数，可通过激活专用覆盖项修改。动态 RMS 无需生成新的静态校准统计；已有权重/Hessian 可以复用，原 LDLQ 没有额外针对激活误差重新优化。任何尺度调参应使用独立校准/验证数据。

码字搜索使用 FP32，共享非持久化搜索表，默认每批最多 4096 个 8D 向量，避免一次构建整个序列的搜索得分矩阵。搜索表不随 `model.half()` 降精度，也不重复保存到每层的检查点；实际激活输出恢复输入 dtype。搜索会增加运行时与中间显存，此版用于验证精度。

### 配置与入口

配置写入 `config.quip_params['activation_quantization']`，旧模型无此字段即关闭。公共配置包括 `bits`、`group_size`、`chunk_size`、`scale_override`、`resid_scale_override`；`quantizer='e8_rvq'`、`backend='qdq'`。新 CLI 同时接受短横线和下划线拼写：

| 参数 | 默认/含义 |
| --- | --- |
| `--activation-bits` | 新量化任务默认关闭；评测/导出未指定时保留模型配置；0/2/3/4 |
| `--activation-group-size` | 默认 128；0=每 token；其余须为 8 的正整数倍 |
| `--activation-chunk-size` | 默认 4096，单次码本搜索的最大 8D 向量数 |
| `--activation-scale` | 覆盖激活归一化 RMS rho；正有限数 |
| `--activation-resid-scale` | 覆盖激活 RVQ 残差尺度；正有限数 |

`quantize_finetune_llama.py` 接受上述配置，启用时要求 `--ft_epochs 0`、Hadamard；`hfize_llama.py` 保留或覆盖该配置；`eval/eval_ppl.py` 可在加载已有 HF QuIP# 模型后覆盖激活设置，例如：

```bash
PYTHONPATH=. ../.venv/bin/python eval/eval_ppl.py \
  --hf_path /path/to/hfized-quip-model --seqlen 4096 \
  --no_use_cuda_graph --activation-bits 4 --activation-group-size 128
```

Python 中可在加载模型后、前向之前调用 `configure_activation_quantization(model, {'bits': 4})`；传 `{'bits': 0}` 关闭。不支持梯度微调/STE、缓存旋转后权重的 `train_mode`、CUDA Graph 或 `torch.compile`。代码明确拒绝梯度输入、`train_mode` 组合和公共加载器的 CUDA Graph 组合；应使用 eager/no-grad 前向。

现有 `reproduce_incoherence.py` 与 `--stage bit-comparison` 仍记录历史 weight-only 的还原权重 PPL，不能用于验证激活量化。新增 `--stage activation-comparison` 使用原生压缩线性层执行 PPL；不会重新量化权重，结果追加到已有 `quantization-profile.json` 的 `activation_comparison`，日志沿用原文件：

```bash
# 现有 Hadamard W2 检查点；完整 WikiText2，context=4096；本机使用较大的搜索批次。
PYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py \
  --stage activation-comparison --weight-bits 2 --activation-bits 4 \
  --context 4096 --activation-chunk-size 65536 \
  >> ../reproduction/quantization-profile.log 2>&1

# 最小端到端检查；短区块 PPL 不能与完整 WikiText2 PPL 比较。
PYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py \
  --stage activation-comparison --weight-bits 2 --activation-bits 4 \
  --context 128 --decode-steps 4 --ppl-blocks 1 \
  >> ../reproduction/quantization-profile.log 2>&1
```

`--activation-bits 0` 测同一路径的 weight-only 基线。W3/W4 需通过 `--checkpoint-dir` 指定实际对应的 128 个矩阵检查点，历史实验的临时 W3/W4 文件已清理。该阶段与其他历史结果分开保存配置/测量范围，检查到外部 GPU 计算进程时等待；记录测评后争用检查，不能完全排除检查间的短暂争用。现有 `PackedForward` profiler 已增加独立 `activation_quantization` 区间（含尺度统计、码字选择及反量化），历史在线结果没有重新测量。

搜索批次是执行参数：本机在 `[4096, 4096]` FP16 激活上核对 A4、128 通道尺度组，`chunk_size=4096/65536` 的 QDQ 输出逐位一致；128-token 端到端检查的 PPL 也一致。单张量 QDQ 分别约 0.478/0.158 秒；这项检查不代表完整模型加速比。完整测试采用 65536，以减少分批调用开销；默认仍为 4096，可按显存选择。

W3/W4 提供自动重建临时权重并测同权重激活配置的阶段（默认 A16/A4）：

```bash
PYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py \
  --stage activation-bit-sweep --comparison-bits 3 4 \
  --activation-chunk-size 65536 --context 4096
```

该阶段复用已有 Hessian，在 `/dev/shm` 重建 128 个矩阵，测完当前位宽即清理；结果写入既有 `quantization-profile.json` 的 `weight_activation_ppl`。默认测 A16/A4；可用 `--comparison-activation-bits 3` 补测 A3，或指定 `0 2 3 4`，其中 `0` 表示 A16。相同配置的完整结果会跳过，新增激活位宽保留原结果和原离线计时。若需重建已清理的权重且已有 A16 结果，先完整复测 A16，要求 PPL 与原对照之差不超过 1e-6 且数据 SHA256 一致，再开始新增配置。重建耗时单独记录在 `checkpoint_rebuilds`。

`--stage activation-baseline --context 4096` 用相同的 8-token 预热和 `evaluate` 计时边界重测 FP16 基线，结果写入同一 JSON 的 `activation_baseline`，保留 `baseline/metrics.json` 历史记录。

本次三项补测的复跑命令（完整配置相同的已完成结果会跳过）：

```bash
# W2A2；改为 --activation-bits 3 可复跑 W2A3。
PYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py \
  --stage activation-comparison --weight-bits 2 --activation-bits 2 \
  --context 4096 --activation-chunk-size 65536

# W3A3；必要时重建权重并校验完整 A16 对照。
PYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py \
  --stage activation-bit-sweep --comparison-bits 3 --comparison-activation-bits 3 \
  --context 4096 --activation-chunk-size 65536
```


### 验证

`quantize_llama/validate_activation.py` 含 CPU 数学/边界检查和可选原生 CUDA 集成检查。CPU 检查覆盖关闭时恒等、全零输入、末组 RMS、token 独立、尺度等变、搜索分批上限、参数拒绝和 FP32 搜索表。GPU 检查覆盖全部 9 种 W/A 组合的单/多 token、旋转后参考计算、关闭时逐位一致、小型 Llama 保存重载与 KV-cache decode。运行：

```bash
PYTHONPATH=. QUIP_ACTIVATION_CUDA_TEST=1 ../.venv/bin/python \
  quantize_llama/validate_activation.py -v
```

2026-10-22，本机 8 项检查通过；小型 Llama 保存/重载后在相同 FP16 buffer 与 SDPA 设置下输出一致。Llama2-7B W2+A0/A2/A3/A4 均完成第一个 128-token WikiText2 区块检查。Hadamard W2/W3/W4 A16 与 A4 已全部完成 context=4096 的完整 WikiText2 测试，每项 83 个区块、339885 个计分 token：A16 PPL 分别为 8.223384 / 5.604442 / 5.268132，A4 PPL 分别为 8.976383 / 5.805925 / 5.423203，均使用原生压缩线性层。十种配置（含 FP16）的统一大表及时间口径见 [RESULTS.md](../reproduction/RESULTS.md) 与 [TIMING_ANALYSIS.md](../reproduction/TIMING_ANALYSIS.md)。2026-10-23 补测完整 W2A2/W2A3/W3A3，PPL 分别为 46.935159 / 12.084731 / 6.373193；W3 重建通过完整 A16 PPL 对照校验，并按相同预热/计时口径补测 FP16。W3A2/W4A2/W4A3 尚未完整测评；当前默认尺度未针对真实激活调优。短区块或单张量 QDQ 时间不能作为完整模型速度结论。

## 5. 误差补偿与精度恢复策略

这些机制发生在不同阶段，可以组合使用，不能统称为微调。

| 策略 | 如何减小误差 | 控制与状态 |
| --- | --- | --- |
| Hessian 感知 Block LDLQ | 对输入二阶统计做 block LDL 分解；按码字列组逆序量化，将已产生的权重误差通过 L 补偿到后续组，优化 Hessian 加权重建误差 | `LDLQ`、`LDLQ_buffered`、`LDLQ_buffered_lowmem`；默认 buffered，`--no_use_buffered` 选朴素版，`--lowmem_ldlq` 选低内存版；FP64 也选 lowmem。三者是同类补偿算法的执行变体 |
| 逐块局部搜索 | 初次 BlockLDLQ 后，基于当前残差和 Hessian H 逐块重新选择码字；区别于初次量化使用 L 的误差反馈 | `--quip_tune_iters` 默认 10；设 0 省略后续逐块局部搜索，仍有 LDLQ 初次误差反馈；不做反向传播 |
| RVQ 残差编码 | 第一阶段码本量化后，再为残差选择第二阶段码字，以额外存储位数减小误差 | 3 bit 为 2+1，4 bit 为 2+2；由 `--codebook` 决定，`--resid_scale_override` 可覆盖残差缩放 |
| 层内微调 / MSE 重建 | 逐步替换线性层后，用校准集拟合 decoder 层的浮点输出；训练已量化层 SU/SV、norm 及当时仍可训练的浮点参数 | `finetune_decoder_layer`，主 CLI 默认 `--ft_epochs 5`；设 0 关闭梯度训练。打包索引不作为梯度参数更新 |
| 端到端微调 / 教师 logits 蒸馏 | 用原浮点模型 softmax 概率作为目标，以 soft-target cross entropy 调整量化模型，缓解层间误差累积 | `finetune_e2e_llama.py`、`finetune_susv_e2e`；训练 SU/SV、norm 和 lm-head 等可训练参数；本机尚未做完整模型微调验证 |
| 低秩修正（遗留） | 量化前用 SVD 去掉低秩分量，量化后对 Hessian 加权残差做 SVD，保存 A/B 修正 | `low_rank_preprocess`、`low_rank_process` 与 `--lora_rank` 尚在；但 `QuantizedLinear.__init__` 明确 `assert rank == 0`，注释说明已移除支持。不能作为当前完整模型主流程已支持的功能，也不是已接通的 LoRA/QLoRA 训练 |

名称来源：逐块局部搜索对应当前代码的 `quip_tune_iters` 循环。前作 [QuIP](https://arxiv.org/abs/2307.13304) §4.2 与附录 B.2 用 Greedy local search / Greedy Updates 解释了 LDLQ 后的坐标下降思想；QuIP# 论文没有单独说明这一步或默认 10 轮。它与梯度微调是独立机制。

辅助的误差控制包括：RHT/RFFT 非相干化以降低异常大坐标的影响、可选 `--rescale_WH` 对角重缩放、每矩阵 RMS/码本尺度、Hessian 正则 `--sigma_reg`。这些属于预处理或数值稳定机制；`show_metrics` 输出 Frobenius/proxy error 只做诊断，不会修复权重。`model/mistral.py` 中的 outlier-channel-split 参数为接口不兼容的遗留内容，不能列为当前已可用的异常通道补偿方案。

## 6. QuIP# 3 bit / 4 bit

**提供，且不只是文档声明。** 三种码本均注册了量化类和推理类，每个向量覆盖 8 个权重；2/3/4 bit 分别对应每向量 16/24/32 个索引位。

| 权重位数 | `--codebook` | 编码方案 | 在线计算与验证状态 |
| --- | --- | --- | --- |
| 2 bit | `E8P12` | 单个 2 bit E8P 码本 | 单行融合 decode-matvec；多行解码后 GEMM；本机已做 Llama-2-7B 完整量化/PPL及在线 profiling |
| 3 bit | `E8P12RVQ3B` | 2 bit E8P + 1 bit E8 残差 | 有打包、解码和推理；单行残差路径填充到 8 行并调用 `lookupmatmul_e81b_k8`，缺少专门高效的 1 bit matvec，不能默认比 4 bit 快；本机已完成 Llama-2-7B Hadamard、no FT 的全模型量化、WikiText2 PPL 及在线 profiling |
| 4 bit | `E8P12RVQ4B` | 两个 2 bit E8P 码本，第二个编码残差 | 单行调用两次 E8P decode-matvec 再合并，多行合并两次解码结果后 GEMM；本机已完成 Llama-2-7B Hadamard、no FT 的全模型量化、WikiText2 PPL 及在线 profiling |

2 bit 上游建议 `--scale_override 0.9`；3/4 bit 建议先使用码本默认 scale 和 residual scale。上述位数选项属于原标准 Hadamard 路径，不代表本地 Fourier 复现脚本已经支持 3/4 bit。也没有注册可独立选择的完整 W1 方案，3 bit 的 1 bit 码本仅作残差级。

## 7. 支持的模型

必须区分算法可迁移、README 发布过检查点、当前入口能加载，以及本机完整验证四种层次。

| 模型/架构 | 当前证据与支持边界 |
| --- | --- |
| Llama 1：7B/13B/30B/65B | 原 README 列有 2/3/4 bit 检查点；当前量化、HF 导出、量化加载主路径均针对 Llama，尚未在本机逐个复核 |
| Llama 2：7B/13B/70B 及对应 chat | 原 README 列有 2/3/4 bit 检查点；项目模型实现 GQA。本机完整验证覆盖 7B/13B Hadamard W2/W3/W4 A16、W2A2/A3/A4、W3A3/A4、W4A4，以及 7B Fourier W2、no FT；70B 尚未在本机完整验证 |
| Mistral 7B、OpenHermes 2.5 | 原 README 列有 2/3/4 bit 检查点，存在 `model/mistral.py`。但当前 `model_from_hf_path` 对压缩模型只分派 `model_type == 'llama'`；HF 导出也硬编码 Llama。Mistral MLP 还向现有 `QuantizedLinear` 传递已不存在的 `outlier_channel_split` 参数。属于遗留适配，不能称为当前开箱即用；模块可导入不代表模型可构建 |
| Llama 3/3.1、Qwen、OPT、Falcon、Mixtral 等 | 没有相应的完整量化/导出适配及本机验证；部分结构可能复用，但需逐项适配和检查。数据工具中出现模型名或 `AutoModelForCausalLM` 不构成量化支持证据 |
| 其他 HF 全精度 causal LM | 通用加载器对非 QuIP# 检查点走 HF AutoModel，部分评测可复用；这不等于可以直接用现有脚本量化该架构 |

另外，码字打包、buffered LDLQ、Hadamard 因子和 CUDA 内核对维度有各自约束；新模型即使同属 Llama，也要核对形状、bias、位置编码和 Transformers 接口。本项目不是任意架构的自动量化转换器。

## 8. 支持的测评与诊断

| 测评/工具 | 覆盖内容 | 边界 |
| --- | --- | --- |
| `eval/eval_ppl.py` | 默认依次计算 WikiText2 和 C4 PPL，默认上下文 4096，固定区块 next-token CE，无 KV cache | 数据集列表写死为两者，CLI 没有 dataset 选项；支持关闭 CUDA Graph；不能用其总耗时冒充逐 token decode 延迟 |
| `eval/eval_zeroshot.py` | 接入 `lm_eval`，`--tasks` 指定任务，`--num_fewshot` 默认为 0，可改为 few-shot，支持结果 JSON | 示例包括 `arc_challenge,arc_easy,boolq,piqa,winogrande`；其他任务取决于安装版本的注册和数据可用性。当前 requirements 固定 `lm_eval==0.3.0`，不能直接套用新版任务清单；此入口实际关闭 CUDA Graph |
| `eval/eval_speed.py` | 固定输入形状的重复前向均值；可设 batch、seqlen、samples，默认 seqlen=1 | 每次均 `use_cache=False`；不是有长前缀 KV cache 的自回归 decode 基准，也没有离线阶段分项 |
| `eval/interactive_gen.py` | 交互/流式采样生成、decode throughput；StaticCache，支持 `--compile` | 是生成演示，含采样/回调等开销；不是独立质量基准，且依赖缓存/编译接口版本 |
| `quantize_llama/reproduce_incoherence.py` | 本地新增：Llama-2-7B 2 bit、no FT，RHT/RFFT 与 FP16 基线的完整 WikiText2 PPL | 评测前还原全部 FP16 权重，测量量化质量；不直接测压缩在线推理。Hessian 仓库等配置针对本次 7B 复现 |
| `quantize_llama/profile_incoherence.py` | 本地新增：离线代表性四矩阵阶段分解；完整模型加载、prefill、真实 KV cache decode，旋转/解码/线性计算/Attention/KV 分项 | 代表性 RHT/RFFT 配置为 E8P12 2 bit；另有 `--stage bit-comparison` 完成 Hadamard 2/3/4 bit 全模型量化、PPL 和在线对比；CUDA events 为 stream 间隔，含 launch 空隙与插桩开销；融合 decode-matvec 不强拆 |
| `quantize_llama/validate_activation.py` | 本地新增：E8/RVQ 激活 QDQ 的 CPU 边界与 CUDA 原生集成验证 | 8 项检查；GPU 测试须显式设置 `QUIP_ACTIVATION_CUDA_TEST=1`，应在空闲 GPU 上执行 |
| `quantize_llama/validate_incoherence.py` | 本地新增：变换往返/范数、Hessian 对称性、proxy loss 不变性和量化后还原检查 | 数学/算子一致性验证，不是语言模型任务精度 |
| `show_metrics`、`scripts/parse_proxy.py` | 权重 Frobenius error、Hessian 加权 proxy error 及日志解析 | Proxy error 不等于 PPL、任务准确率或推理速度 |
| `quiptools/test_*.py`、`benchmark_e8p.py` | 低层 CUDA 测试/实验代码 | 部分脚本保留旧接口；不能将文件存在等同于已适配当前 API 的完整性能基准 |

`lib/utils/gptq_data_utils.py` 还包含 PTB、PTB-new 和 C4-new 的数据工具，但 `get_test_tokens` 仅分派 WikiText2/C4/C4-new，标准 PPL 入口只调用 WikiText2/C4。因此 PTB 属于可复用 loader，当前没有直接接通的 PTB PPL 命令。`LMEvalAdaptor` 中的 T5/OPT 分支也不构成对应架构的量化适配；OPT 适配器还明确抛出 `NotImplementedError`。

模型文件中有 sequence-classification/QA 类，当前项目没有配套完整的量化训练与测评入口，不能据此认定这些任务已验证。类似地，未发现现成的多模态、长上下文质量套件或量化鲁棒性专用评测入口。

## 9. 已验证范围与当前缺口

已有本机记录见工作区本地的 `ENVIRONMENT.md` 和 `smoke_test.log`：2/3/4 bit 随机小矩阵量化及 batch=1/3 推理均通过；小型单层量化 Llama 的 prefill/KV decode、FlashAttention 前反向通过。入口 `--help` 通过只代表能解析参数，不代表完整工作流通过。2026-10-30 重跑激活测试：6 项 CPU 检查通过，2 项可选 CUDA 检查跳过；此次没有重跑 GPU 测评。

完整实验保存在工作区本地的 `reproduction/RESULTS.md` 和 `reproduction/TIMING_ANALYSIS.md`：Llama-2-7B 2 bit、no FT 的 RHT/RFFT 完整量化与还原 FP16 后的 WikiText2 PPL 已完成，并另测完整压缩模型在线耗时。Hadamard 3/4 bit 全模型量化、WikiText2 PPL 和在线 profiling 也已完成；7B/13B 的上述 weight-only/weight-activation 配置均已有完整 WikiText2 记录。尚未完成梯度微调、Mistral、C4 和零样本任务的本机端到端验证。

最初功能核查通过无模型小调用确认 Kronecker 缺少 `torch` 导入，未修复该遗留分支。2026-10-22 新增了第 4 节的激活 E8/RVQ QDQ 参考实现及验证。后续 Fourier 通用量化/导出仍需旋转类型元数据及前向分派；Mistral 需要修正加载/导出与线性层接口；激活压缩存储、专用低比特计算与 KV cache 量化仍需独立实现。
