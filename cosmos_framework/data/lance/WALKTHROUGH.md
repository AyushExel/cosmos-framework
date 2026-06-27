# Implementation walkthrough — the LanceDB Cosmos dataloaders

A code-level tour of *what we changed and why*, written so you can read it once and then
explain it to someone else. It covers, per loader: **how cosmos stores the data today**,
**the schema we propose for Lance**, **what the offline converter moves out of the hot
path**, and **how the runtime loader reproduces the base sample exactly**. The shared
runtime machinery (Permutation API, worker-safe handles, batched fetch, plain-binary vs
blob) is at the end.

For *why each one is faster* and the measured numbers, see [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)
and [`BENCHMARKS.md`](BENCHMARKS.md). For *correctness*, the equivalence tests in
`tests/data/lance/`. This file is about the **code**.

---

## The one idea

Cosmos's three training dataloaders each re-do, *every sample of every epoch*, work whose
result never changes: decode + resize + concatenate three robot camera views (action),
decode + resize a clip with a fresh ffmpeg process (vision-SFT), or walk a sequential tar
/ HF stream with no random access (VLM). The base format forces it — raw per-view mp4s, a
source clip at native resolution, a tar shard.

The proposal is the same in all three cases:

> **Do the per-epoch transform once, offline, and store the result as one columnar row
> per training sample in a Lance table — media bytes inline, plus the minimal metadata
> needed to reconstruct the exact sample.** The training loader then does *random-access,
> batched, object-store-native* reads of small rows and a much lighter decode.

That single move buys two different things: a **lighter hot path** (the "representation"
win, action + vision-SFT) and a **better access layer** (random access + global shuffle +
native `s3://`, the "access" win, all three — and the whole point on object storage).

---

## 1. Action / DROID — `LanceDROIDComposedDataset`

### How cosmos stores it today
LeRobot v3: per-view mp4s under `videos/<view>/chunk-*/file-*.mp4`, with *all episodes of a
view concatenated into one file* (in the 327-episode set, **three ~0.5 GB mega-files**, one
per view — wrist + 2 exteriors). Tabular state/action/timestamps live in parquet under
`data/`. `DROIDLeRobotDataset.__getitem__` (`droid_lerobot_dataset.py:228`) for every
sample: seeks **three** views, decodes a window from each (torchcodec), `F.interpolate`s the
two exteriors to half-res, and concatenates into one `(3, T, 270, 320)` tensor
(`_load_concat_video`, `droid_lerobot_dataset.py:288`). Local filesystem only — there is no
S3 reader.

### Proposed Lance schema — one row per **episode**
Built by `tools/lance_datagen/build_composed_droid.py`:

```python
pa.schema([
    pa.field("episode_index", pa.int64()),   # which episode this clip is
    pa.field("ep_start",      pa.int64()),    # its first global frame index
    pa.field("length",        pa.int64()),    # frame count
    pa.field("video_bytes",   pa.large_binary()),  # the COMPOSED 270x320 clip, mp4
])
```

`video_bytes` is the base's **exact** resize+concat output (`_load_concat_video` is called
in the converter, `build_composed_droid.py:94`), re-encoded **all-intra** (`gop=1`). The
3-view → 1-clip fusion at half-res means the store is **0.35× the original on disk**, not a
blowup. The parquet index (state/action/timestamps) is **left untouched** — the Lance
loader still reads it via the inherited base `__init__`, so all pose/action math is shared.

### What the converter moves offline
The three-stream decode + `F.interpolate` + `torch.cat`. Done once, per episode, instead of
once per sample per epoch.

### What the runtime loader does (`action_dataset.py:257`)
`LanceDROIDComposedDataset(DROIDLeRobotDataset)` **inherits all indexing/pose/action logic**
and overrides only the video path:
- maps each flat sample index → `(episode, frame offset)` with the base's `_valid_cum` /
  `_ep_starts` arrays (unchanged),
- decodes **one** small composed clip (no interpolate, no concat — baked in) with a
  per-worker LRU `VideoDecoder` cache keyed by episode (`_ensure_decoders`,
  `action_dataset.py:328`), `seek_mode="approximate"` (exact at `gop=1`, skips the
  full-file index scan),
- batches the whole DataLoader batch in `__getitems__` (`action_dataset.py:352`): group the
  needed frames per clip → **one `get_frames_at` per clip**, then assemble the result dict
  through the base's `_build_result` (so idle-frame, normalization, caption logic is shared).

`LanceDROIDComposedIterable` (`action_dataset.py:402`) pairs it with episode-shuffle (the
production shuffle): windows of an episode stream contiguously, so each clip is fetched +
decoded once and reused across all its windows.

A second class, `LanceDROIDDataset` (`action_dataset.py:57`), stores the **original** mp4
bytes (no re-encode) in a `*_videos` blob table — used by the equivalence test to prove
**bit-exact** video. The composed clip differs only by the H.264 re-encode (PSNR ~32 dB).

---

## 2. Vision-SFT / Bridge — `LanceVisionSFTDataset`

### How cosmos stores it today
`SFTDataset` (`sft_dataset.py:97`) is an `IterableDataset`: a JSONL manifest lists per-clip
`vision_path` + caption windows; **each sample is a separate source mp4** (default on S3).
`process_one_sample` (`sft_dataset.py:178`) downloads the bytes (`download_from_s3`, which
also reads local), spawns an **ffmpeg subprocess** to decode the window with a `scale`
filter (resize to training resolution), center-crops, temporally truncates, and tokenizes
the caption — all per sample, every epoch.

### Proposed Lance schema — one row per **clip/window**
Built by `tools/lance_datagen/build_vision_sft.py`:

```python
pa.schema([
    pa.field("clip_id", pa.string()),
    pa.field("width",   pa.int64()), pa.field("height", pa.int64()),   # original size
    pa.field("start_frame", pa.int64()), pa.field("end_frame", pa.int64()),
    pa.field("temporal_interval", pa.int64()),
    pa.field("enc_h", pa.int64()), pa.field("enc_w", pa.int64()),      # stored resized size
    pa.field("fps",   pa.float64()),
    pa.field("caption_json", pa.string()),   # structured caption (verbatim) or ""
    pa.field("caption",      pa.string()),   # dense backup
    pa.field("video_bytes",  pa.large_binary()),  # clip PRE-RESIZED to training res, mp4
])
```

`video_bytes` is the clip decoded once and **resized with the base's exact `scale_hw`**
(`build_vision_sft.py:130`), re-encoded all-intra. The spatial center-crop is intentionally
**left to decode time** (so the stored clip is a clean rectangle); the window math +
truncation + crop are re-applied at runtime. The caption is stored verbatim → token-ids are
**exact**.

### What the converter moves offline
The per-sample ffmpeg process spawn + full-resolution decode + on-the-fly resize.

### What the runtime loader does (`vision_sft_dataset.py:53`)
A **map-style** `Dataset` (random access, unlike the base's iterable):
- in-process torchcodec decode (no subprocess) of a clip **already at training resolution**
  → far fewer pixels, no resize,
- the **same** `entire_chunk` window math + center-crop + temporal truncation as the base
  (`_window_plan`, `vision_sft_dataset.py:192` — deliberately mirrors
  `SFTDataset.process_one_sample`),
- caption selection reused from the base via `select_caption`, then the same cosmos
  `tokenize_caption`,
- batched `__getitems__` (`vision_sft_dataset.py:231`): one `get_frames_at` per clip,
  per-worker decoder LRU cache.

This is the largest single-modality win and the only one that holds **end-to-end** (the
only non-video work is one cheap tokenize, so the video saving isn't masked).

---

## 3. VLM / LLaVA-OneVision — `LanceVLMDataset` + `LanceVLMShuffleScan`

### How cosmos stores it today
The shipped source factory `get_llava_ov_streaming` (`configs/base/vlm/experiment/
llava_ov_vlm.py:79`) streams `lmms-lab/LLaVA-OneVision-Data` from the HuggingFace Hub
(`streaming=True`) — **sequential shard reads, a bounded shuffle buffer** (approximate, not
global), re-streamed every epoch, **no random access**. Cosmos has no local/S3 VLM source.

### Proposed Lance schema — one row per **sample**
Built by `convert_llava_to_lance` (`vlm_dataset.py:72`):

```python
pa.schema([
    pa.field("sample_id",     pa.string()),
    pa.field("image_bytes",   pa.large_binary()),  # ORIGINAL encoded bytes (no re-encode)
    pa.field("conversations", pa.string()),         # ShareGPT JSON, verbatim
])
```

No transcode — the original PNG/JPEG bytes are stored as-is, so records are **byte-identical**
to the HF stream and the downstream processor produces identical tensors. This is purely an
**access-layer** change (no representation win, and none claimed).

### What the runtime loader does (`vlm_dataset.py:89`)
Two access modes over the same table:
- **`LanceVLMDataset`** — map-style **O(1) random access** via the Permutation API → **true
  global shuffle** (shuffle row indices, `take` them). Yields the exact
  `{id, image, conversations}` dict `get_llava_ov_streaming` yields, so the processor chain
  is unchanged.
- **`LanceVLMShuffleScan`** (`vlm_dataset.py:143`) — the right pattern for object storage:
  shuffle *fragment order* + a row buffer over a sequential columnar scan
  (`to_batches(..., batch_readahead=8)`) → **bandwidth-bound** reads on S3 with
  WebDataset-quality shuffle, but columnar (much faster than tar) and with random access
  still available.

Honest caveat (in code + docs): the VLM **end-to-end** step is gated by the Qwen
image-processor, so the big raw-access win shows e2e only at scale / on object storage. VLM
is never the combined-mixer bottleneck.

---

## 4. Shared runtime machinery (all three loaders)

### 4a. Worker-safe Permutation pattern
Following `lerobot-lancedb` and the `training/object-detection` reference: the `Dataset`
stores only **connection params**; `__getstate__` nulls every live handle (lance dataset,
`Permutation`, decoder cache) so it pickles cleanly to **spawn** workers (Lance is **not
fork-safe** — always `multiprocessing_context="spawn"`). Each worker lazily reopens its own
handles in `_ensure_open` on first access. See `action_dataset.py:84`,
`vision_sft_dataset.py:105`, `vlm_dataset.py:112`.

### 4b. Batched `__getitems__` is the hot path
PyTorch's `DataLoader` hands the **whole batch's indices at once** to `__getitems__`. Every
loader groups by file/clip and issues **one `take`/`take_blobs` + one `get_frames_at` per
file**, not per sample — large contiguous reads + decodes instead of N tiny ones.

### 4c. Plain `large_binary` vs blob-v2 — the S3 read knob
`take_blobs` (lance blob-v2) returns lazy `BlobFile` handles read **one GET at a time** in
Python → serialized, latency-bound on S3. For clips <2 MB, storing the bytes as a **plain
`large_binary`** column and reading via `ds.take(indices, columns=[...])` lets Lance
parallelize the GETs across the IO thread pool (**~6× faster on S3**, measured). The loaders
**auto-detect** the encoding (`_is_blob` from the column's `lance-encoding:blob` metadata,
`action_dataset.py:307` / `vision_sft_dataset.py:121`) and pick `take` vs `take_blobs`;
converters default to `--storage plain`. Bytes are identical either way, so equivalence
holds. `data_storage_version` stays at the stable **2.1** (2.2 is unstable in Lance 7.0.0).

### 4d. Memory: freeing the base's dead per-frame index
`ActionBaseDataset.__init__` (`base_dataset.py`) materializes `self._rows` — **one Python dict per
frame** (its own comment notes ~tens of GB at full DROID's ~18M frames). The DROID loaders read windows
from compact numpy arrays via `_window_rows` and override `__len__`, so `self._rows` is **dead weight**
for them — and with `spawn` workers (Lance needs spawn) it's pickled into *every* worker, multiplying the
footprint. The Lance action loaders free it in `__init__` (`_FreeBaseRowsMixin`, `action_dataset.py`),
which is safe (unused) and leaves output unchanged. Effect: per-worker RAM grows ~1.4 KB/frame for the base
vs ~0.09 KB/frame for Lance, so at real DROID scale Lance is several× lighter per worker (the base trends to
OOM). The base can't drop it globally — other action loaders (agibot/robomind/bridge) genuinely use
`self._rows` — so the per-loader Lance port is the clean place to fix it. Numbers: [`BENCHMARKS.md`](BENCHMARKS.md) §3.

### 4e. Why equivalence is preserved by construction
- **Action**: all index/pose/action/caption logic is *inherited unchanged* from
  `DROIDLeRobotDataset`; only the video bytes' origin differs. Video is bit-exact for the
  raw-bytes variant, within H.264 tolerance for the composed variant.
- **Vision-SFT**: the runtime re-applies the base's exact window/crop/truncate math and the
  same `tokenize_caption`; captions are stored verbatim → **token-ids exact**, video within
  re-encode tolerance.
- **VLM**: original bytes + conversations stored verbatim → **records byte-identical**.

Proven in `tests/data/lance/` (action 8/8 bit-exact, vision-SFT 7/7 token-exact, VLM 3/3
byte-identical).

---

## 5. How to pitch it upstream (the one-paragraph version)

> Cosmos's action / vision-SFT / VLM loaders each redo a fixed per-epoch transform and read
> a format that forbids random access and native object-store reads. Porting each to a
> single Lance table — one columnar row per training sample, media bytes inline, the
> per-epoch transform precomputed offline — gives a lighter decode hot path (action,
> vision-SFT), true random access + global shuffle (all three), and native `s3://` training
> with selective/parallel reads the base can't do without a full download or FUSE mount. The
> loaders subclass / mirror the shipped classes so output is verified equivalent
> (bit/token-exact), making them a drop-in swap. The only storage-format proposal is "inline
> media + minimal metadata, columnar, plain `large_binary`"; everything else is reuse.
