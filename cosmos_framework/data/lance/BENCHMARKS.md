# Benchmarks — LanceDB vs base Cosmos dataloaders

Single node (48 CPU + NVIDIA L40S), 327 DROID episodes, batch 16, **CPU decode on both
sides** (the base can only decode on CPU), RAW (no model) unless a row says otherwise. Lance
tables use **plain `large_binary`** storage. Every base side is the **genuine shipped loader**
(see §6); mechanisms in [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md), code in [`WALKTHROUGH.md`](WALKTHROUGH.md),
reproduce in [`REPRODUCE.md`](REPRODUCE.md).

Two regimes:
- **LOCAL** — all loaders read local disk (cosmos's "pre-download then train" workflow).
- **S3** — Lance reads `s3://` natively; the base reaches S3 the only way it can: action via
  `S3DROIDLeRobotDataset` (a benchmark standin that downloads the per-view mega-mp4s then runs
  the genuine decode — see `benchmarks/lance/base_standins.py`), vision-SFT via the genuine
  `SFTDataset`'s own per-sample boto3 download, VLM via HF-Hub streaming (its only mode).

> **Fairness note.** Earlier revisions compared against reconstructions / an s3fs FUSE mount,
> which *inflated* the S3 speedups. These numbers use the genuine base classes throughout, so
> they are lower than (and supersede) prior drafts — and honest.

---

## 1. Combined 3-loader throughput (samples/s)

The 1:1:1 mixer is gated by the slowest loader. Workers are action/vlm/vsft per-loader.

| comparison | LOCAL | S3 |
| ---------- | ----- | -- |
| base 4/4/4 vs lance 4/4/4 (cosmos default workers) | 92.6 → 254.8 (**2.75×**) | 72.6 → 265.2 (**3.65×**) |
| base 18/4/18 vs lance 18/4/18 (tuned workers) | 280.1 → 931.0 (**3.32×**) | 251.7 → 1240.7 (**4.93×**) |
| **base 4/4/4 (as-shipped) vs lance 18/4/18 (tuned)** | **10.1×** | **17.1×** |

Two compounding wins: the **Lance loaders** and **per-loader worker rebalancing** (cosmos ships
a flat ~4 workers/loader and never rebalances toward the bottleneck — its "multiplex" is
ratio-based modality mixing, not worker allocation). The bottom row is the real out-of-the-box
delta. Reproduce: `benchmarks/lance/run_matrix.sh` (each cell a separate
`bench_combined_faithful.py --trios …`). Full-S3 lance (1240.7) > LOCAL lance (931.0) at tuned
workers because S3 reads run on the async IO-thread pool and don't steal decode CPU.

---

## 2. Single-loader (per-modality) throughput

base → lance (speedup), 18 workers for action/vsft, 4 for VLM.

| loader (recipe) | LOCAL | S3 |
| --------------- | ----- | -- |
| action / DROID (`action_policy_droid`), episode-shuffle | 172.4 → 304.9 (**1.77×**) | 152.6 → 295.7 (**1.94×**) |
| vision-SFT / Bridge (`vision_sft_nano`), raw | 157.5 → 780.6 (**4.96×**) @18w; 62.5 → 462.2 (**7.40×**) @8w | 130.1 → 1047.3 (**8.05×**) @18w; 42.3 → 370.3 (**8.76×**) @8w |
| VLM / LLaVA (`llava_ov`), raw access | 518 → 17,939 (**~35×**) | 518 → 35,741 (**~69×**) |

Notes:
- **action** is already the most optimized base loader (3-view decode), so the fair delta is
  modest (~1.8×); the win is the pre-composed 1-clip representation, not NVDEC.
- **vision-SFT** is the biggest e2e win and it **holds end-to-end** (~7.5×, `--mode e2e`,
  60.8 → 453.9 @8w) because its only non-video work is a cheap tokenize. The 18-worker LOCAL
  ratio (4.96×) is lower than 8-worker (7.40×) because the genuine `SFTDataset` base streams
  sequentially and parallelizes well; both are honest — quote the worker count.
- **VLM** raw ratio is large because the base is cosmos's *shipped* `streaming=True` source
  (`get_llava_ov_streaming`), which decodes via the streaming protocol (≈518 samples/s even
  across its 5 shards). This is **access-layer only**: VLM **end-to-end is ≈1×** (Qwen
  image-processor bound) and VLM is **never the combined-mixer bottleneck**. Don't quote it as
  a training speedup.

---

## 3. Memory (action / DROID) — the scaling story

Throughput is half the scaling problem; the base is also memory-heavy. `bench_memory.py`,
8 workers, action loader, **PSS** (proportional set size — the fair physical-RAM metric, since
fork shares pages copy-on-write and RSS would double-count them).

### 3a. At the 327-ep benchmark scale (96k frames) — base is *leaner*

| | peak total PSS | per-worker PSS |
| - | -------------- | -------------- |
| base | 6.8 GB | 651 MB |
| Lance | 8.8 GB | 737 MB |

At toy scale Lance is ~+90 MB/worker heavier — each `spawn` worker pays a fixed ~290 MB
Arrow/Lance runtime (memory pools + IO thread pool) that outweighs the data savings. (Flat
across decoder-cache sizes 2→32, so it is the runtime, not the cache.)

### 3b. Scaling with dataset size — Lance wins, and the gap widens

The base's `ActionBaseDataset.__init__` materializes `self._rows` = **one Python dict per
frame** (its own comment notes ~tens of GB at full DROID's ~18M frames). For the DROID loader
this is **dead weight** (it reads windows from compact numpy arrays via `_window_rows` and
overrides `__len__`), and with `spawn` workers it is pickled into **every** worker. The Lance
loaders free it (`_FreeBaseRowsMixin`; output unchanged, equivalence still passes). Measured by
replicating the subset N× (`build_scaled_droid.py`), random-sampler iteration, 8 workers:

| dataset | frames | base per-worker PSS | Lance per-worker PSS | peak total: base / Lance |
| ------- | ------ | ------------------- | -------------------- | ------------------------ |
| 1× | 96k | 651 MB | 737 MB | 6.8 / 8.8 GB |
| 4× | 385k | 1044 MB | 777 MB | 10.9 / 9.2 GB |
| 16× | 1.54M | **2612 MB** | **863 MB** | **27.1 / 10.9 GB** |

base per-worker grows ~**1.4 KB/frame** (the `_rows` dict list — 128 / 533 / 2153 MB at
1×/4×/16×); Lance grows ~**0.09 KB/frame** (compact arrays only). **Crossover ≈ 4× (~1,300
episodes)**: below it the base is leaner, above it Lance is. At 16× Lance per-worker is **~3×
lower** and peak total **~2.5× lower**, and it keeps widening — extrapolated to full DROID
(~18M frames) the base is ~25 GB/worker (× workers → OOM) while Lance stays a few GB. **Lance is
the one that scales.** Disk footprint is also 0.35× (§5).

### 3c. fork (copy-on-write) cuts both
Switching DataLoader workers to `fork` (lance fork support is experimental but produced
**identical** output here) shares the parent's pages COW: e.g. Lance peak PSS 8.8 → 5.1 GB at
1×. Helps both sides; use the same start method on both for an apples-to-apples comparison.

---

## 4. Correctness (prerequisite for any throughput/memory claim)

| loader | per-sample test | batch test (`__getitems__` hot path) |
| ------ | --------------- | ------------------------------------ |
| action / DROID | `test_action_equivalence.py` **8/8** — raw-bytes video bit-exact, action bit-exact | `test_batch_equivalence.py` — composed (benchmarked): labels bit-exact, video within H.264 tol; raw-bytes: pixel-identical |
| vision-SFT | `test_vision_sft_equivalence.py` **7/7** — token-ids exact, video within tol | batch: token-ids exact, video within tol |
| VLM | `test_vlm_equivalence.py` **3/3** — records byte-identical | batch: records byte-identical |

22 tests total. Labels / token-ids / VLM records are **exact**; the fast video paths (composed
action, vision-SFT) match within one offline H.264 re-encode. Plain-vs-blob storage is
byte-identical, so equivalence holds for both encodings.

---

## 5. Storage — plain `large_binary`, and dataset sizes

`take_blobs` (blob-v2) returns lazy handles read one GET at a time → serialized on S3; storing
clips as **plain `large_binary`** and reading via columnar `take` parallelizes across the IO
thread pool (**~6× faster** S3 reads for <2 MB clips). Loaders auto-detect; converters default
to `--storage plain`. `data_storage_version` stays at **2.1** (2.2 is unstable in Lance 7.0.0).

On-disk size of the combined store actually built (Lance tables, gop=1 all-intra):

| modality | base format & size | Lance | ratio |
| -------- | ------------------ | ----- | ----- |
| action / DROID, 327 eps | raw 3-view mp4 **1.55 GB** | composed **0.55 GB** | **0.35×** |
| VLM / LLaVA figureqa, 99,995 | HF parquet **2.22 GB** | **2.23 GB** | ~1.0× (original bytes, no re-encode) |
| vision-SFT / Bridge, 200 clips | raw mp4 + jsonl **0.10 GB** | **0.11 GB** | ~1.1× |
| **combined** | **~3.9 GB** | **~2.9 GB** | **0.75×** |

Smaller overall, driven by the composed action store (3→1 view, half-res > the all-intra
penalty). The bit-exact action variant (raw mp4 bytes) is ~1.5 GB ≈ base.

---

## 6. Base loader storage (verified in the cosmos source)

- **action / LeRobot** — local filesystem only, `Path(root)` (`data/vfm/action/datasets/base_dataset.py`).
- **VLM / LLaVA** — HF-Hub streaming, `get_llava_ov_streaming` → `load_dataset(..., streaming=True)`
  (`configs/base/vlm/experiment/llava_ov_vlm.py`).
- **vision-SFT** — S3 via boto3 `download_from_s3` with a local fallback (`sft_dataset.py`, `helper.py`).

So real cosmos training reads local disk **and** remote object storage at once; the benchmarks
exercise both regimes against these genuine loaders.

---

## Reproduce
Env + datasets + conversions + the exact commands: [`REPRODUCE.md`](REPRODUCE.md). In short:
`source benchmarks/lance/.venv-gpu/bin/activate`, build the plain tables with
`tools/lance_datagen/*`, then `bash benchmarks/lance/run_matrix.sh` (throughput) and
`python benchmarks/lance/bench_memory.py …` (+ `build_scaled_droid.py` for the scaling rows).
