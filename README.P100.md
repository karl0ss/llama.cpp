# llama.cpp P100 / sm_60 port

This branch carries a curated port of the
[shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches)
patch set onto current llama.cpp `master`, targeted at the Tesla P100
(GP100, compute capability 6.0) as a single-GPU decode device.

Base: llama.cpp master `eb182d85c` ("place d2t by its set_rows use…").
The port is four commits on top; every change that is not
behavior-neutral is off by default behind an environment variable.

## Why

`llama.cpp` upstream optimizes for modern NVIDIA hardware (Volta+). Pascal
GP100 lacks DP4A, byte-wise SIMD compares, and modern tensor cores, so Q-vector
dot products, quantized matmuls and flash attention all fall onto slow fallback
paths. shinbunbun's patch set addresses that; this branch keeps the working
subset against a much newer llama.cpp base (the set was written against an
older tree) so it can keep up with master instead of aging in place.

## What is ported

| Patch | Title | Status / notes |
|---|---|---|
| 01 | vmad dp4a sm60 | ported — PTX `vmad` scalar-MAC replaces SW byte dot products on sm_60 |
| 02 | mmvq rows-per-block sm60 | ported — 4 rows/block for MUL_MAT_ID generic; measured optimum on GP100 |
| 04 | concat non-cont flat | ported |
| 05 | mmvf f32 pascal | ported (to `mmq-config-pascal-dp4a.cuh`, upstream renamed the file) |
| 07-09 | mmvq moe / mmid-batch / nwarps-small-k sm60 | ported |
| 10 | mmvq q8_1 activation cache | ported — gate/up pairs quantify the activation once |
| 11 | penalties direct | ported |
| 12 | mmvq f16 sm60 | ported (`mmvq-f16-sm60.cu/.cuh`) |
| 13 | sampler prefilter | ported (`common/sampling.cpp`, pf fields carried through clone) |
| 14 | getrows narrow rows | ported |
| 15 | mtp draft vocab | ported |
| 16 | cpy fastdiv | ported |
| 17 | norm register cache | **dropped** — upstream reworked rms_norm/l2_norm kernels (float4 templates); the double-read issue is superseded. The fused pre-add/l2-norm launch machinery from 19/20 is ported anew on the upstream kernels. |
| 18 | fuse sibling nodes | ported |
| 19 | fuse pre-add rms norm | ported — `rms_norm_f32` gained a `do_pre_add` template mode + `ggml_cuda_op_rms_norm_fused_pre_add()`, dispatched under `GGML_CUDA_FUSE_PRE_ADD` |
| 20 | fuse add unary mul | ported (`GGML_CUDA_FUSE_ADD_UNARY_MUL`) |
| 21 | sched reset lazy | ported |
| 22 | decode sched slots | ported — per-decode-slot schedulers in `llama-context.cpp`, `dec_slots[]` LRU, integrated with upstream's `gf_res_prev`/graph-reuse rework |
| 31 | fattn f16 kv chunk | ported — FA over f16-sourced KV processed in 16k chunks (`GGML_CUDA_FA_F16_KV_CHUNK`), with the chunk loop gated `!use_sparse` so sparse-FA kernels are unaffected |
| 32 | sched split prefetch | ported (`LLAMA_SCHED_PREFETCH`) against upstream's refactored copy loop — `ggml_backend_sched_copy_input` two-pass host-weight ordering is preserved, prefetched inputs are skipped |
| 34 | mul-mat-id negative ids | ported — CPU `mul_mat_id` treats a negative id as "row contributes nothing" |
| 36 | mmvq chunk large batch | ported (`GGML_CUDA_MMVQ_CHUNK_MIN_MIB`) — chunk a small-column quantized matmul through MMVQ instead of expanding the whole weight for cuBLAS |
| 37 | getrows narrow batched | ported |
| 38 | getrows q4_0 block | ported |
| 40 | mmq iq4_nl threads | ported into the Pascal MMQ tile table (256→128 threads/block) |

Not ported: 39 (GDN lanes — needs the GDN kernel lineage of patches 23-27),
33 (sched weight prefetch), 41-46 (sparse-attention / qwen4exp / expert
residency — model-specific).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_FUSE_PRE_ADD` | 0 | fuse residual ADD into RMS_NORM prologue (needs 18) |
| `GGML_CUDA_FUSE_ADD_UNARY_MUL` | 0 | fuse ADD→UNARY→MUL chains |
| `GGML_CUDA_FA_F16_KV_CHUNK` | 16384 | f16-KV FA chunk size; ≤0 disables |
| `LLAMA_SCHED_PREFETCH` | off | overlap GPU tail with next CPU split (measured -5% on our GLM-4.7 workload — leave off unless your CPU/GPU balance differs) |
| `GGML_CUDA_MMVQ_CHUNK_MIN_MIB` | -1 (off) | MMVQ column-chunking threshold in MiB |
| `GGML_CUDA_DISABLE_FUSE_L2_NORM` | 0 | disable the l2-norm launch fusion |

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build -j
```

## Measurements

On a Tesla P100 16GB (Driver 580.x / CUDA 13.0, one GPU + AVX2 CPU), the
round-1 subset (patches 01-22+31) measured +25-35% prefill and decode vs the
same llama.cpp base without the port, on interleaved A/B benches with
GLM-4.7-Flash (MoE, MLA, ncmoe offload) and gpt-oss-20b as the workload
models, and round 2 (32-40) was kept behavior-neutral-or-gain on the same
method.

Benching discipline note (this matters): single-run numbers on this class of
box are noisy to the point of uselessness. Use interleaved A/B
(laguna→new→laguna→new, same prompt) and compare medians.

The round-3 subset (28, 27k) was ported, measured and parked on a separate
local branch — our measurements showed decode regressions on this config, so
it is not in this branch.

## Credits

All patch content is shinbunbun's from the linked repo. This branch is the
re-base, conflict resolution and porting work for those patches against newer
llama.cpp — see the individual commits for per-hunk decisions.