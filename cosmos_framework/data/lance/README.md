# LanceDB-powered Cosmos dataloaders

Drop-in LanceDB replacements for the three dataloaders Cosmos mixes during training
(LeRobot **action**, WebDataset/HF **VLM**, local **vision-SFT**), built for higher
dataloading throughput **and better memory scaling**, while preserving the training signal.
Output is verified equivalent to the base loaders, so they're a faithful swap.

- **How each speedup works** (per-loader mechanisms): [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)
- **Code + schema walkthrough** (what changed, per loader): [`WALKTHROUGH.md`](WALKTHROUGH.md)
- **Full numbers** (throughput, memory, scaling, correctness): [`BENCHMARKS.md`](BENCHMARKS.md)
- **Reproduce from scratch**: [`REPRODUCE.md`](REPRODUCE.md)

All comparisons are against the **genuine shipped base loaders** (CPU decode both sides); where
the base can't reach S3 natively, an explicit benchmark standin in `benchmarks/lance/base_standins.py`
gives it the fairest possible object-store path (documented there).

## Headline — throughput

Combined 3-loader throughput, 327 DROID episodes, batch 16, RAW (workers = action/vlm/vsft):

| comparison | LOCAL | S3 |
| ---------- | ----- | -- |
| base 4/4/4 vs lance 4/4/4 (cosmos default) | **2.75×** | **3.65×** |
| base 18/4/18 vs lance 18/4/18 (tuned) | **3.32×** | **4.93×** |
| **base 4/4/4 (as-shipped) vs lance 18/4/18 (tuned)** | **10.1×** | **17.1×** |

Two compounding wins: the **Lance loaders** and **per-loader worker rebalancing** (cosmos ships a
flat ~4 workers/loader and never rebalances toward the bottleneck). Per-loader (single-modality):
action **1.8×**, vision-SFT **~5–8×** (holds end-to-end), VLM raw-access large but **≈1× e2e**
(image-processor bound; never the mixer bottleneck). Full tables: [`BENCHMARKS.md`](BENCHMARKS.md).

## Headline — memory scaling

The base also has a memory-scaling problem: `ActionBaseDataset.__init__` materializes a
**per-frame Python-dict index** (`self._rows`, dead weight for DROID) that `spawn` workers each
copy — ~tens of GB at full DROID. The Lance loaders free it. Action loader, per-worker PSS:

| dataset | base | Lance |
| ------- | ---- | ----- |
| 96k frames (327 eps) | 651 MB | 737 MB |
| 1.54M frames (16×) | **2612 MB** | **863 MB** |

At toy scale Lance is slightly heavier (fixed Arrow runtime); past ~1,300 episodes it wins, and
at real scale it's **~3× lower per worker** (the base trends to OOM). Plus 0.35× on disk. Details +
fork/COW: [`BENCHMARKS.md`](BENCHMARKS.md) §3.

## What changed, per loader (mechanisms in [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md), code in [`WALKTHROUGH.md`](WALKTHROUGH.md))

- **Action / LeRobot** — `action_dataset.py`. Base decodes 3 camera views + resizes + concats per
  sample every epoch. `LanceDROIDComposedDataset` serves **one pre-composed, pre-resized, all-intra
  clip per episode** + a per-worker decoder cache. Labels bit-exact; video within H.264 re-encode
  tolerance. A bit-exact `LanceDROIDDataset` variant stores raw mp4 bytes for strict parity.
- **VLM** — `vlm_dataset.py`. Base streams HF-Hub / tar with a bounded shuffle buffer, no random
  access. `LanceVLMDataset` gives O(1) random access + true global shuffle; `LanceVLMShuffleScan` is
  the object-storage chunked-shuffle pattern. Records byte-identical.
- **Local vision-SFT** — `vision_sft_dataset.py`. Base spawns ffmpeg per sample to decode+resize.
  `LanceVisionSFTDataset` decodes a **pre-resized, all-intra per-clip** stream in-process. Token-ids
  exact; the win holds **end-to-end** (~7.5×).

Storage: clips are **plain `large_binary`** (not blob-v2) — ~6× faster columnar S3 reads for <2 MB
payloads; loaders auto-detect, converters default to `--storage plain`.

## Correctness
Action **8/8 bit-exact**, vision-SFT **7/7** (token-ids exact), VLM **3/3** (byte-identical), plus a
batch-level `test_batch_equivalence.py` covering the benchmarked composed loader via `__getitems__`
— `tests/data/lance/`. Throughput/memory claims are only meaningful because the output matches.

## Why this isn't practical without LanceDB
- **Object-store-native**: the stock action/VLM loaders read the local FS or stream from HF; Lance
  reads `s3://` natively with selective/parallel reads — no full download or FUSE mount.
- **Structural**: true random access + global shuffle (a tar/stream is sequential; its shuffle is an
  approximate buffer), plus columnar/filtered reads.
- **The representation + memory wins** require an indexed, versioned, object-store-native, shuffle-
  sampled multimodal store of pre-transformed clips — i.e. you'd be rebuilding Lance.

## Reproduce
Full recipe (env, downloads, conversions, S3, expected numbers): [`REPRODUCE.md`](REPRODUCE.md).
Quick: Python 3.12, matched `torch`/`torchvision`/`torchcodec` + `nvidia-npp-cu12` on
`LD_LIBRARY_PATH`, `lancedb`/`pylance`, `transformers`, `datasets`, `boto3`, system `ffmpeg`.
**`source benchmarks/lance/.venv-gpu/bin/activate`**. Datasets are public on HF
(`lerobot/droid_1.0.1`, `lmms-lab/LLaVA-OneVision-Data`, `nvidia/BridgeData2-Subset-Synthetic-Captions`).

```bash
python tools/lance_datagen/build_composed_droid.py --root <droid>/success --uri <lance> --gop 1 --storage plain
python tools/lance_datagen/build_vision_sft.py --jsonl <bridge>/.../video_dataset_file.jsonl --uri <lance> --storage plain
pytest tests/data/lance/                 # equivalence (set the *_LANCE_URI / *_JSONL env vars)
bash benchmarks/lance/run_matrix.sh      # LOCAL / S3 × {4/4/4, tuned} × {base, lance}
python benchmarks/lance/bench_memory.py --side lance --root <droid>/success --uri <lance> --random
```

Layout: dataloaders in `cosmos_framework/data/lance/`, converters in `tools/lance_datagen/`,
benchmarks in `benchmarks/lance/`, equivalence tests in `tests/data/lance/`.
