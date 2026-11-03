# Models running on this P100, with launch commands and speed vs stock

This is the model line-up I run every day on one Tesla P100, served through
[llama-swap](https://github.com/mostlygeek/llama-swap). It lists each model's exact
launch command and measured speed against **stock upstream llama.cpp at the same
commit**, so the numbers show what this fork adds and nothing else. Details of the
patches themselves are in [README.P100.md](README.P100.md).

Measured 2026-11-03.

## Hardware

| | |
|---|---|
| GPU | Tesla P100-PCIE-16GB (sm_60, HBM2), PCIe gen3 x16, driver 580.142, CUDA 12.9 |
| CPU | Xeon E5-2680 v4, 14 cores / 28 threads, Broadwell, **AVX2 only (no AVX-512)** |
| RAM | 94 GB DDR4 |
| Shared | The P100 also runs Frigate (NVR object detection, ~1.8 GB VRAM baseline), so every model fits in about 14.4 GB |

## Line-up

| llama-swap name | Model / quant | Context | Engine | Role |
|---|---|---|---|---|
| `glm-4.7-flash-64k` | GLM-4.7-Flash UD-Q3_K_XL (MoE, MLA) | 64k | **this fork** | daily coding model |
| `nemotron-3-nano-30b-q3xl-131k` | Nemotron-3-Nano-30B-A3B UD-Q3_K_XL (hybrid Mamba MoE) | 131k | **this fork** | agentic |
| `ornith-1.5-9b-128k` | Ornith-1.5-9B Q8_0 (gated-delta-net hybrid) + vision mmproj | 128k | **this fork** | general + images |
| `qwen3.5-4b-3d-64k` | Qwen3.5-4B-3D-Coder Q8_0 (GDN hybrid) | 64k | **this fork** | fast small coder |
| `gpt-oss-20b-code-128k` / `-fast-8k` | gpt-oss-20b Q4_K_M | 128k / 8k | older-base P100 build ("laguna") | fastest model overall |
| `gemma4-12b-fast-16k` | Gemma-4-12B-it UD-Q5_K_XL | 16k | older-base P100 build ("laguna") | general |
| `qwen3.8-flash-next-128k` | Qwen3.8-Flash-Next 125B MoE, GSQ-RCO Q2_0 | 128k | [Strata](https://github.com/jmnargi/Strata-V100), sm_60 build | largest model; experts in RAM |
| `bonsai2-27b-262k` | Ternary Bonsai 2 27B PQ2_0 | 262k | PrismML llama.cpp fork | ternary weights |
| `maple-20b-cpu-131k` | Maple 20B-A1B TQ2_0 | 131k | ternary llama.cpp fork, **CPU only** | always resident, used for memory extraction |

Only one GPU model is loaded at a time. Maple runs on the CPU and stays resident alongside
it.

## Speed vs stock

The benchmark is `llama-bench` with each model's **live flags** (listed below). There are
two interleaved rounds per build, and each cell is their mean. "Stock" is upstream
llama.cpp `28b812d6c` built with the same CMake flags. "Fork" is this repo's `master`
(the same commit plus the P100 stack). pp = prompt processing; tg = decode at the given
KV depth. All numbers are tokens/s.

| Model | Test | Stock | Fork / live build | Change |
|---|---|---|---|---|
| **GLM-4.7-Flash** | pp2048 | 475 | 449 | −5% |
| | pp16384 | 207 | **327** | **+57%** |
| | tg128 @ 0 | 37.4 | 38.0 | +2% |
| | tg128 @ 16k | 24.6 | 24.9 ¹ | +1% |
| **Qwen3.5-4B-3D-Coder** | pp2048 | 1316 | 1370 | +4% |
| | pp16384 | 1067 | **1186** | **+11%** |
| | tg128 @ 0 | 51.7 | **64.0** | **+24%** |
| | tg128 @ 16k | 43.6 | **55.0** | **+26%** |
| **Ornith-1.5-9B** | pp2048 | 655 | 611 ² | −7% |
| | pp16384 | 578 | 601 | +4% |
| | tg128 @ 0 | 32.3 | **38.0** | **+18%** |
| | tg128 @ 16k | 29.3 | **33.9** | **+16%** |
| **Nemotron-3-Nano-30B** | pp2048 | 563 | 503 | **−11%** ³ |
| | pp16384 | 519 | 483 | −7% |
| | tg128 @ 0 | 40.9 | **48.5** | **+18%** |
| | tg128 @ 16k | 38.0 | **43.2** | **+14%** |
| **gpt-oss-20b** (laguna, live) | pp2048 | 572 | **860** | **+50%** |
| | pp16384 | 482 | **718** | **+49%** |
| | tg128 @ 0 | 57.9 | **85.7** | **+48%** |
| | tg128 @ 16k | 48.6 | **61.3** | **+26%** |
| gpt-oss-20b on fork `master` | pp2048 / pp16384 | 572 / 482 | 567 / 484 | parity |
| | tg128 @ 0 / @ 16k | 57.9 / 48.6 | 80.7 / 61.1 | +39% / +26% |
| **Gemma-4-12B** (laguna, live) | pp2048 | 331 | 289 | −13% |
| | pp8192 | 296 | 280 | −5% |
| | tg128 @ 0 | 19.5 | **22.9** | **+18%** |
| | tg128 @ 8k | 18.2 | **21.7** | **+19%** |

Notes:

1. **GLM runs with `GGML_CUDA_FA_F16_KV_CHUNK=0`.** Patch 31's f16-KV chunking splits
   attention into two passes once the context passes 16k. On GLM's 576-wide MLA heads that
   costs **−17% decode** (20.5 t/s with chunking at 16k depth; 15.3–16.1 vs 17.9–19.1 t/s
   at 29k on the live server). GLM's KV cache is small, so full f16 conversion costs only
   ~75 MB at 64k. The other models showed no difference either way, so only the GLM entry
   sets it. The 24.9 in the table is the chunking-off figure.
2. Ornith pp2048 is noisy: round 1 was −1%, round 2 −12%.
3. **Known regression:** Nemotron prompt speed is 7–11% below stock, and decode is 14–18%
   above it. The cause is somewhere in the bundled shinbunbun port commit (patches 01-22)
   and has not been bisected yet. For agentic use the decode gain outweighs it.

**Why gpt-oss and Gemma use "laguna":** laguna is the same shinbunbun port on an older
upstream base (late September 2026). Upstream later lost about a third of its Pascal prompt
speed on gpt-oss. Laguna is 50% faster on prompts than both stock and the fork, and decodes
at the same speed as the fork. On Gemma, laguna trades a little prompt speed for +18–19%
decode. Both stay on laguna until upstream's Pascal prefill recovers.

**Long prompts on the live server** (llama-swap, full context preallocated, prompt cache
off, compared with the previous fork build before GEMM flash attention):

| Model | 30k-token prompt before | After |
|---|---|---|
| GLM-4.7-Flash (64k) | 127 t/s | **242 t/s** (+91%) |
| Qwen3.5-4B (64k) | 900 t/s | **1086 t/s** (+21%) |
| Ornith-9B (128k) | 467 t/s | **526 t/s** (+12%) |
| Nemotron-30B (131k) | 488 t/s | 509 t/s (+4%) |

## Launch commands

`$MODELS` is the GGUF directory. `$PORT` is filled in by llama-swap. Every entry binds
to `127.0.0.1` behind llama-swap.

### GLM-4.7-Flash, 64k (this fork)

```sh
env GGML_CUDA_FUSE_PRE_ADD=1 GGML_CUDA_FUSE_ADD_UNARY_MUL=1 GGML_CUDA_FA_F16_KV_CHUNK=0 \
  llama-server -m $MODELS/GLM-4.7-Flash-UD-Q3_K_XL.gguf --host 127.0.0.1 --port $PORT \
  -c 65536 -np 1 -ngl 999 -ncmoe 12 --load-mode none -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 4096 -t 16 --jinja --temp 0.6 --top-p 0.95 --top-k 20 \
  --cont-batching --metrics --no-warmup
```

- `-ncmoe 12` puts 12 layers of experts in system RAM. That is the setting that fits 64k
  of q8_0 KV next to Frigate.
- q8_0 KV: q4_0 KV measured *slower* on this MLA architecture.
- `-b/-ub 4096` measured +20% prompt speed over 1024/2048.
- The two `FUSE` env vars measured +2.3% decode and are deterministic.

### Nemotron-3-Nano-30B-A3B, 131k (this fork)

```sh
llama-server -m $MODELS/Nemotron-3-Nano-30B-A3B-UD-Q3_K_XL.gguf --host 127.0.0.1 --port $PORT \
  -c 131072 -np 1 -ngl 999 -ncmoe 22 --load-mode none -fa on -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 2048 -t 16 --jinja --temp 0.6 --top-p 0.95 --top-k 20 \
  --cont-batching --metrics --no-warmup
```

Hybrid Mamba: the KV cache is only ~3.4 KB/token, so 131k fits. `-ub 2048` measured +36%
prompt speed over 1024. `-ub 4096` runs out of memory.

### Ornith-1.5-9B, 128k, with vision (this fork)

```sh
llama-server -m $MODELS/Ornith-1.5-9B-Q8_0.gguf --host 127.0.0.1 --port $PORT \
  -c 131072 -np 1 -ngl 999 -fa on -b 1024 -ub 512 -t 16 --cache-ram 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 2 \
  --mmproj $MODELS/mmproj-Ornith-1.5-9B-BF16.gguf --no-mmproj-offload \
  --jinja --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
  --cont-batching --metrics --no-warmup --timeout 1200
```

- The vision tower runs on the CPU (`--no-mmproj-offload`) because the P100 has no room
  left for it.
- MTP drafting (`draft-mtp`, 2 tokens) is on. Server decode speed varies with draft
  acceptance, so measure decode with plain `llama-bench -d`.

### Qwen3.5-4B-3D-Coder, 64k (this fork)

```sh
llama-server -m $MODELS/Qwen3.5-4B-3D-Coder-Q8_0.gguf --host 127.0.0.1 --port $PORT \
  -c 65536 -np 1 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 16 --cache-ram 4096 \
  --jinja --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.7 --top-p 0.95 --top-k 40 --repeat-penalty 1.0 \
  --cont-batching --metrics --no-warmup --timeout 1200
```

Thinking is disabled because it ran away on this fine-tune.

### gpt-oss-20b (laguna build)

```sh
# 128k coding entry
llama-server -m $MODELS/gpt-oss-20b-Q4_K_M.gguf --host 127.0.0.1 --port $PORT \
  -c 131072 -np 1 -ngl 999 -fa on -b 2048 -ub 1024 -t 16 --cache-ram 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --chat-template-kwargs '{"reasoning_effort":"low"}' \
  --temp 0.3 --top-p 0.9 --top-k 40 --repeat-penalty 1.05 --cont-batching --metrics --no-warmup --timeout 1200
# 8k fast entry: -c 8192 -b 4096 -ub 2048, f16 KV (-ub 2048 measured +42% prompt speed)
```

### Gemma-4-12B, 16k (laguna build)

```sh
llama-server -m $MODELS/gemma-4-12b-it-UD-Q5_K_XL.gguf --host 127.0.0.1 --port $PORT \
  -c 16384 -np 1 -ngl 48 -fa on -b 2048 -ub 1024 -t 16 --cache-ram 4096 \
  --jinja --temp 0.7 --top-p 0.95 --top-k 20 --repeat-penalty 1.05 --cont-batching --metrics
```

### Other engines (not llama.cpp mainline)

| Entry | Engine | Measured on this box |
|---|---|---|
| Qwen3.8-Flash-Next 125B (Q2_0) | [Strata-V100](https://github.com/jmnargi/Strata-V100) built with `-DSTRATA_EXPERIMENTAL_SM60=ON`, native pack + GGUF experts, `--expert-cache auto --spec 4 --mtp`, 128k with `--kv-grow` | **30–36 t/s decode**, ~274 t/s prompt, ~48 s cold load; the CPU expert pool is the bottleneck (no AVX-512) |
| Ternary Bonsai 2 27B (PQ2_0) | PrismML llama.cpp fork, `-c 262144 -ctk q4_0 -ctv q4_0 -ub 1024` | PQ2_0 is the only fast ternary path on the P100 (+52% decode vs Q3_K_XL); ~143 t/s prompt measured on an earlier 128k entry at `-ub 2048` |
| Maple 20B-A1B (TQ2_0) | ternary llama.cpp fork, CPU only, `taskset -c 0-13 -t 14` | **56 t/s decode with zero GPU**; pinning to 14 cores gave +8% |

## Method

- **Interleaved A/B** (old → new → old → new, same inputs): single runs on this box vary by
  ±20%, mostly from Frigate and VMs sharing the host.
- **Every prompt-speed gain is checked with `llama-perplexity --kl-divergence`** before it
  ships. An MoE→cuBLAS change looked like +13% and turned out to raise perplexity by 55%
  (see [README.P100.md](README.P100.md#measured-dead-ends)).
- The stock build is upstream `28b812d6c`, built with `-DCMAKE_CUDA_ARCHITECTURES=60
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON`.
