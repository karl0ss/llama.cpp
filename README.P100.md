# llama.cpp P100 / sm_60 port

**`master` on this fork is upstream llama.cpp plus the P100 stack, rebased
regularly (expect force-pushes).** Upstream's own master is at
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

Upstream llama.cpp, tuned for the Tesla P100 (GP100, compute capability 6.0)
as a single-GPU inference card. It contains:

1. A curated port of the
   [shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches)
   set onto current llama.cpp `master`.
2. **cuBLAS-GEMM flash attention for long prompts** (from
   [Kmic-68's fork](https://github.com/Kmic-68/llama.cpp)), extended here to
   MLA models (GLM-4.7-Flash, deepseek2) and made VRAM-bounded.

Each change that alters numerical results or memory behaviour can be turned off
with an environment variable (see [Environment variables](#environment-variables)).

## Branches

| Branch | Base | Adds |
|---|---|---|
| `p100-ported` (2026-10-21) | master `eb182d85c` | shinbunbun rounds 1+2 (01-22, 31, 32, 34, 36-38, 40) |
| `p100r4` (2026-10-29) | master `0893ec98f` | the same stack rebased. Upstream CUDA MoE gate/up+GLU MMQ fusion (#31188) and top-k fusion (#31288) come with the newer base. Adds patches 23 + 47 |
| `p100r5-fagemm` (2026-11-03) | `p100r4` | GEMM flash attention for long prompts, with MLA 576/512 support and query-token slicing |
| **`master`** (2026-11-03) | upstream `28b812d6c` | the full stack above, rebased onto current upstream; this is the branch to build |

## Headline results (Tesla P100 16 GB)

All numbers come from interleaved A/B runs on one box: the binaries alternate on
the same prompt, over at least 2 rounds. Single runs on this machine vary by
±20% and are not used.

| Workload | Metric | Before | After | Change | From |
|---|---|---|---|---|---|
| gpt-oss-20b Q4_K_M | decode | 62.6 t/s | 79.6 t/s | **+27%** | shinbunbun port (vs vanilla at the same base) |
| GLM-4.7-Flash Q3_K_XL | decode | 34.3 t/s | 37.1 t/s | **+8%** | shinbunbun port |
| Ornith-1.5-9B Q8_0 | decode | 53.6 t/s | 55.5 t/s | **+3.5%** | patch 23 (GDN fusion) + newer base |
| GLM-4.7-Flash, 64k server | 8k-token prompt | 326 / 313 t/s | 419 / 402 t/s | **+29%** | GEMM flash attention |
| GLM-4.7-Flash, 64k server | 30k-token prompt | 127 / 125 t/s | 242 / 239 t/s | **+91%** | GEMM flash attention |
| GLM-4.7-Flash | decode at 8k depth | 29.3-30.1 t/s | 29.5-29.8 t/s | parity | GEMM path does not run during decode |
| Qwen3.5-4B-3D-Coder Q8_0, 64k server | 30k-token prompt | 912 / 888 t/s | 1085 / 1087 t/s | **+21%** | GEMM flash attention |
| Ornith-1.5-9B Q8_0, 128k server | 30k-token prompt | 450 / 483 t/s | 530 / 523 t/s | **+12%** | GEMM flash attention |
| Nemotron-3-Nano-30B-A3B, 131k server | 30k-token prompt | 494 / 482 t/s | 500 / 518 t/s | +2-7% | GEMM flash attention (few attention layers) |
| Ornith-1.5-9B | decode at 8k depth (no MTP) | 33.6-35.9 t/s | 35.5-36.6 t/s | parity | |

The hybrid models (Qwen3.5, Ornith, Nemotron) have few attention layers, so
the GEMM path only helps them on long prompts. At 8k their prompt speed is
within noise. With MTP drafting on, decode speed through the server varies
from run to run because draft acceptance varies; check decode with plain
`llama-bench -d`.

The GLM server rows used the production command line: `-c 65536 -ngl 999
-ncmoe 12 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -t 16`, with the two fusion env
vars set and prompt caching disabled. VRAM use was identical on both builds
(15,189 MiB).

## GEMM flash attention for long prompts

**Problem.** The P100 has no tensor cores, so flash attention runs on the generic
`flash_attn_tile` kernel. That kernel reaches only a fraction of the card's 2:1
fp16 rate. MLA models are the worst case: GLM-4.7-Flash attends with K=576 and
V=512 head sizes over 20 query heads per KV head. In an 8k-token prompt,
`flash_attn_tile<576,512,4,4>` was **69% of all GPU time** (nsys), and prompt
speed fell from 545 t/s at 2k tokens to 356 t/s at 8k. Upstream confirms
the Pascal/MLA behaviour in
[ggml-org/llama.cpp#19020](https://github.com/ggml-org/llama.cpp/issues/19020).
Turning flash attention off fixes prompt speed but makes decode worse at depth.

**Fix.** For batches of 128 or more query tokens against 4096 or more KV positions
on pre-Volta GPUs, attention runs as cuBLAS fp16 GEMMs instead:
`S = KᵀQ` per 2048-key chunk, an online-softmax kernel, then `O += V·P`. The
running output and the softmax state stay in fp32 across chunks, and K/V are
dequantized one chunk at a time. Decode (one query token) and short prompts
still use the tile kernel, so decode speed does not change.

The GEMM path is Kmic-68's work; it reached this branch via qr0n's patch 0010
in [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp). This
branch ports only the GEMM part. The GQA-6 and q4_0 tile-kernel changes in that
patch are left out. Two changes were made here:

- **MLA support.** The original path required K and V to have the same head
  size, which excluded every MLA model. In deepseek2-style graphs V is an
  offset-0 view of the first 512 dims of each 576-dim K row. The gate now
  accepts `K=576, V=512` when `V->data == K->data`, and the per-chunk V
  dequant (row stride `V->nb[1]`, 512 columns from each row start) reads that
  view correctly.
- **Token slicing.** Scratch memory scales with query tokens × heads per KV
  group: about 14 KB per column at chunk 2048, or about **1.2 GB** for
  GLM at `-ub 4096`. That ran out of memory once a 64k cache was allocated.
  Query tokens are now processed in slices of at most `GGML_CUDA_FA_GEMM_COLS`
  tokens×heads (default 20480, about 300 MB). Each slice offsets Q, the mask
  rows and dst, and the output matches unsliced runs exactly (KLD to every
  printed digit). Models with few heads per KV group (Qwen3.8-27B: 6, at
  `-ub 2048`) stay as a single slice, so their behaviour does not change.

**Quality.** Checked with `llama-perplexity --kl-divergence` at 8k context
(GLM-4.7-Flash, tile kernel as the reference):

| Path | PPL | Mean KLD vs tile | Same top token |
|---|---|---|---|
| tile kernel (reference) | 4.563 | — | — |
| upstream FA off (f16 KV) | 4.577 | 0.083 | 91.7% |
| **GEMM fp16 (default)** | 4.595 | 0.092 | 91.5% |
| GEMM fp32 (`GGML_CUDA_FA_GEMM_PREC=32`) | 4.610 | 0.092 | 91.4% |

The GEMM path is as close to the tile kernel as upstream's own FA-off path is.
All PPL differences are within the ±0.127 error bars, and fp32 mode does not
improve quality.

## What is ported from shinbunbun

| Patch | Title | Status / notes |
|---|---|---|
| 01 | vmad dp4a sm60 | ported. PTX `vmad` scalar-MAC replaces software byte dot products on sm_60 |
| 02 | mmvq rows-per-block sm60 | ported. 4 rows/block for generic MUL_MAT_ID, the measured optimum on GP100 |
| 04 | concat non-cont flat | ported |
| 05 | mmvf f32 pascal | ported, into `mmq-config-pascal-dp4a.cuh` (upstream renamed the file) |
| 07-09 | mmvq moe / mmid-batch / nwarps-small-k sm60 | ported |
| 10 | mmvq q8_1 activation cache | ported. Gate/up pairs quantize the activation once |
| 11 | penalties direct | ported |
| 12 | mmvq f16 sm60 | ported (`mmvq-f16-sm60.cu/.cuh`) |
| 13 | sampler prefilter | ported (`common/sampling.cpp`; the prefilter fields are copied when the sampler is cloned) |
| 14 | getrows narrow rows | ported |
| 15 | mtp draft vocab | ported |
| 16 | cpy fastdiv | ported |
| 17 | norm register cache | **dropped**. Upstream reworked the rms_norm/l2_norm kernels, which removed the double read this patch fixed. The fused pre-add/l2-norm launch code from 19/20 is ported onto the new upstream kernels instead. |
| 18 | fuse sibling nodes | ported |
| 19 | fuse pre-add rms norm | ported. `rms_norm_f32` gained a `do_pre_add` mode, used when `GGML_CUDA_FUSE_PRE_ADD` is set |
| 20 | fuse add unary mul | ported (`GGML_CUDA_FUSE_ADD_UNARY_MUL`) |
| 21 | sched reset lazy | ported |
| 22 | decode sched slots | ported. Per-decode-slot schedulers in `llama-context.cpp`, integrated with upstream's graph-reuse rework |
| 23 | fuse gdn beta sigmoid | ported in `p100r4`. Fires on gated-delta-net hybrids (Ornith, Qwen3.5) |
| 31 | fattn f16 kv chunk | ported. FA over f16-sourced KV runs in 16k chunks (`GGML_CUDA_FA_F16_KV_CHUNK`); sparse-FA kernels are unaffected |
| 32 | sched split prefetch | ported (`LLAMA_SCHED_PREFETCH`) onto upstream's refactored copy loop |
| 34 | mul-mat-id negative ids | ported. CPU `mul_mat_id` treats a negative id as a row that contributes nothing |
| 36 | mmvq chunk large batch | ported (`GGML_CUDA_MMVQ_CHUNK_MIN_MIB`) |
| 37 | getrows narrow batched | ported |
| 38 | getrows q4_0 block | ported |
| 40 | mmq iq4_nl threads | ported into the Pascal MMQ tile table (256→128 threads/block) |
| 47 | sched split inputs cap | ported in `p100r4`, off by default (`GGML_SCHED_SPLIT_INPUTS_CAP=1`) |

Not ported: 39 (GDN lanes, reverted in an earlier round), 33 (sched weight
prefetch, measured −3 to −8% decode on GLM), 28 and 27k (decode regressions;
parked on a local branch), and 41-46 (sparse attention, qwen4exp and expert
residency, all model-specific).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_FA_GEMM` | on | GEMM flash attention for long prompts on pre-Volta GPUs; `0` restores upstream behaviour |
| `GGML_CUDA_FA_GEMM_COLS` | 20480 | maximum query tokens × heads per GEMM-FA slice (bounds scratch VRAM) |
| `GGML_CUDA_FA_GEMM_PREC` | 16 | `32` runs the GEMM-FA QKᵀ and PV in fp32 (slower; no quality gain measured) |
| `GGML_CUDA_FUSE_PRE_ADD` | 0 | fuse the residual ADD into the RMS_NORM prologue (needs patch 18) |
| `GGML_CUDA_FUSE_ADD_UNARY_MUL` | 0 | fuse ADD→UNARY→MUL chains |
| `GGML_CUDA_FA_F16_KV_CHUNK` | 16384 | f16-KV flash-attention chunk size; ≤0 disables |
| `LLAMA_SCHED_PREFETCH` | off | overlap the GPU tail with the next CPU split (measured −5% on GLM-4.7; leave off unless your CPU/GPU balance differs) |
| `GGML_CUDA_MMVQ_CHUNK_MIN_MIB` | -1 (off) | MMVQ column-chunking threshold in MiB |
| `GGML_SCHED_SPLIT_INPUTS_CAP` | 0 | patch 47 split-input cap |
| `GGML_CUDA_DISABLE_FUSE_L2_NORM` | 0 | disable the l2-norm launch fusion |

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build -j
```

## Measured dead ends

These were tried on the same box and rejected. They are listed so nobody has to
repeat them.

- **MoE matmuls through fp16 cuBLAS on sm_60.** Sending large-batch
  MUL_MAT_ID to cuBLAS instead of MMQ gave +12-14% prompt speed but **broke
  output**: PPL 8.70 → 13.46 and same-top 85%. fp32 accumulation was just as bad
  (PPL 12.94) and lost the speed. Upstream's choice of MMQ for MoE on Pascal
  is correct.
- **Retuning the 576/512 tile kernel.** Of 7 launch configs (threads,
  occupancy, `nbatch_fa`, `nbatch_K`), upstream's (256, 2, 64, 64) was the
  fastest. A 32-column variant that fits in 48 KB of shared memory was 11%
  slower.
- **ik_llama.cpp** (`2af2f550`, sm_60): prompt and decode speed at parity with
  this branch on GLM-4.7-Flash.
- **Round 3 (28, 27k):** decode regressions; parked.

## Benchmarking discipline

Single-run numbers on this class of box are too noisy to use. Use interleaved
A/B runs (old→new→old→new, same prompt) and compare medians. Check any
prompt-speed gain with `llama-perplexity --kl-divergence` before trusting it: the
MoE/cuBLAS result above looked like a clean +13% until perplexity was measured.

## Credits

- shinbunbun: the P100 patch set
  ([llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches)).
- Kmic-68: cuBLAS-GEMM flash attention for pre-Volta GPUs
  ([fork](https://github.com/Kmic-68/llama.cpp)). qr0n packaged it as patch
  0010 in [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp).
- This branch: the rebases, conflict resolution, MLA support, token slicing
  and measurements. See the individual commits for per-hunk decisions.
