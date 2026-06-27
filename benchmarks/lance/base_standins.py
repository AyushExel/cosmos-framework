# SPDX-License-Identifier: OpenMDW-1.1
"""Benchmark-only standins over the GENUINE Cosmos base loaders.

These exist for ONE reason: so the throughput benchmarks compare the Lance loaders
against the **real shipped base classes** (not reconstructions), including in storage
regimes the base classes don't natively reach. Every class here subclasses a genuine
Cosmos loader and changes ONLY where the bytes come from / how the loader is driven —
never the decode / resize / concat / tokenize hot path, which is inherited verbatim.
They are deliberately kept in ``benchmarks/`` (not the shippable ``cosmos_framework``
package) because they are measurement scaffolding, not library code.

Two storage facts drive the design (both verified against the real data):

* **Action** — ``DROIDLeRobotDataset`` is LOCAL-ONLY (``base_dataset.py`` reads
  ``Path(root)``; there is no S3 reader). The DROID v3 layout stores *all* episodes of
  a view concatenated into **one ~0.5 GB mega-mp4 per view** (3 files total). So the
  only way the base can read from S3 is to **materialize the whole file** — there is no
  partial/selective remote read. :class:`S3DROIDLeRobotDataset` models exactly the
  cosmos-prescribed "pre-download to local, then train" workflow: it pulls the 3 mega
  files from S3 once (logged), then runs the **identical** base decode. Lance, by
  contrast, reads a small per-episode clip remotely with no full mirror.

* **Vision-SFT** — the shipped ``SFTDataset`` is already S3-native AND local (its
  ``download_from_s3`` falls back to ``Path.read_bytes`` for non-``s3://`` paths,
  ``helper.py``). Each sample is a *separate* ~0.5 MB source mp4, so on S3 every sample
  is a genuine remote GET (no amortization) — the real object-store cost. So the SAME
  shipped class is the base for BOTH regimes; we only point ``vision_path`` at local vs
  ``s3://``. :class:`BenchSFTDataset` is a thin driver over it that (a) sets
  ``shard_world_size=1`` so it runs without ``torch.distributed``, (b) builds the same
  Qwen tokenizer the Lance loader uses (so token-ids stay exact) instead of via a hydra
  ``tokenizer_config``, and (c) adds a ``skip_tokenize`` flag for raw video-only timing.
  The per-sample hot path (download → ffprobe → ffmpeg decode+scale → tokenize) is the
  shipped ``process_one_sample``, unchanged.

The VLM base needs no standin: the genuine source factory
``cosmos_framework.configs.base.vlm.experiment.llava_ov_vlm.get_llava_ov_streaming``
is imported and called directly by the benchmarks (it is HuggingFace-Hub streaming in
every regime — cosmos has no local/S3 VLM base).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from cosmos_framework.data.vfm.action.datasets.base_dataset import _MODE_CHOICES  # noqa: F401  (re-export parity)
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import (
    _IMAGE_FEATURES,
    DROIDLeRobotDataset,
)
from cosmos_framework.data.vfm.local_datasets.sft_dataset import (
    SFTDataset,
    _load_sft_metadata_from_s3,
)

_QWEN_TOKENIZER = "Qwen/Qwen2.5-7B"


# ════════════════════════════════════════════════════════════════════════════
# Action — S3 standin over the genuine DROIDLeRobotDataset
# ════════════════════════════════════════════════════════════════════════════
class S3DROIDLeRobotDataset(DROIDLeRobotDataset):
    """Genuine :class:`DROIDLeRobotDataset`, but the per-view mega-mp4s are sourced
    from S3 instead of local disk.

    The base is local-only and the DROID layout is a few ~0.5 GB whole-file blobs, so
    "reading from S3" *means* downloading those whole files — there is no partial read.
    This subclass pre-fetches them once into a stable local cache (keyed by the S3
    prefix, reused across runs) and then defers to the base's exact decode/resize/concat
    path. The tabular index (parquet/meta) is read from the local ``root`` as usual —
    it is cheap metadata, and isolating the *video* read is what we want to measure.

    Steady-state throughput therefore equals the LOCAL base (same files, same decode);
    the S3 cost is the **mandatory full materialization** (logged at construction) plus
    the inability to do selective remote reads — precisely the thing the Lance loader
    removes. Created solely for the S3 action benchmark.
    """

    def __init__(
        self,
        root: str,
        s3_bucket: str,
        s3_prefix: str,
        *,
        region: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, **kwargs)
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix.strip("/")
        self._region = region
        key = self._s3_prefix.replace("/", "_")
        self._cache_root = Path(cache_dir or os.path.join(tempfile.gettempdir(), "_s3base_droid", key))
        self._materialize_from_s3()

    def _rel_for(self, episode: dict[str, Any], video_key: str) -> str:
        """The view's relative video path under root (mirrors the base ``_video_path``)."""
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        return self._info["video_path"].format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx,
            episode_chunk=chunk_idx, episode_file=file_idx,
        )

    def _materialize_from_s3(self) -> None:
        """Download every distinct view mega-file referenced by the episodes (once)."""
        import boto3

        rels: set[str] = set()
        for episode in self._episodes.values():
            for video_key in _IMAGE_FEATURES.values():
                rels.add(self._rel_for(episode, video_key))
        s3 = boto3.client("s3", region_name=self._region) if self._region else boto3.client("s3")
        total_bytes, t0, n_dl = 0, time.perf_counter(), 0
        for rel in sorted(rels):
            dst = self._cache_root / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            key = f"{self._s3_prefix}/{rel}"
            tmp = dst.with_suffix(dst.suffix + f".part{os.getpid()}")
            s3.download_file(self._s3_bucket, key, str(tmp))
            os.replace(tmp, dst)
            total_bytes += dst.stat().st_size
            n_dl += 1
        if n_dl:
            dt = time.perf_counter() - t0
            print(
                f"[S3DROIDLeRobotDataset] materialized {n_dl} mega-file(s) "
                f"({total_bytes / 1e6:.0f} MB) from s3://{self._s3_bucket}/{self._s3_prefix} "
                f"in {dt:.1f}s — base must fully download before it can read.",
                flush=True,
            )

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        # Read the materialized local copy; everything downstream is the genuine base.
        return self._cache_root / self._rel_for(episode, video_key)


# ════════════════════════════════════════════════════════════════════════════
# Vision-SFT — thin driver over the genuine SFTDataset (works local AND s3)
# ════════════════════════════════════════════════════════════════════════════
def _qwen_tokenizer_config():
    """An object the genuine ``SFTDataset.__init__`` instantiates (``instantiate`` passes
    a plain object through) to expose ``.tokenizer`` — the SAME Qwen2.5-7B tokenizer the
    Lance loader uses, so token-ids are identical between base and Lance (the shipped
    ``SFTDataset`` tokenizes via the same ``tokenize_caption`` helper)."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(_QWEN_TOKENIZER))


def load_sft_metadata(jsonl_path: str, *, s3_bucket: str | None = None, s3_prefix: str | None = None,
                      min_frames: int = 61) -> list[dict]:
    """Load the bridge SFT metadata from a LOCAL jsonl (the shipped loader's own parser),
    optionally rewriting each ``vision_path`` to ``s3://`` for the S3 regime."""
    meta = _load_sft_metadata_from_s3(None, jsonl_path, min_frames=min_frames)
    if s3_bucket and s3_prefix:
        base_dir = os.path.dirname(os.path.abspath(jsonl_path))
        pref = s3_prefix.strip("/")
        for m in meta:
            vp = m["vision_path"]
            rel = os.path.relpath(vp, base_dir) if os.path.isabs(vp) or os.path.exists(vp) else vp
            m["vision_path"] = f"s3://{s3_bucket}/{pref}/{rel}"
    return meta


class BenchSFTDataset(SFTDataset):
    """Driver over the GENUINE ``SFTDataset`` for throughput benchmarks (local or S3).

    Inherits the shipped ``process_one_sample`` / ``__iter__`` unchanged; only overrides
    construction (direct Qwen tokenizer, no hydra), single-shard setup (no distributed),
    and adds ``skip_tokenize`` for raw video-only timing. Drive it as an IterableDataset.
    """

    def __init__(
        self,
        metadata: list[dict],
        *,
        num_video_frames: int = 16,
        resolution: str = "256",
        temporal_interval_mode: str = "entire_chunk",
        frame_selection_mode: str = "first",
        temporal_compression_factor: int = 4,
        skip_tokenize: bool = False,
    ) -> None:
        super().__init__(
            metadata=metadata,
            num_video_frames=num_video_frames,
            resolution=resolution,
            s3_credentials={},  # empty -> boto3 default cred chain (~/.aws); unused for local paths
            temporal_interval_mode=temporal_interval_mode,
            frame_selection_mode=frame_selection_mode,
            tokenizer_config=_qwen_tokenizer_config(),
            cfg_dropout_rate=0.0,
            temporal_compression_factor=temporal_compression_factor,
        )
        self.skip_tokenize = bool(skip_tokenize)
        # single-process / single-shard so __iter__ never touches torch.distributed
        self.shard_world_size = 1
        self.shard_rank = 0
        self.shard_id = 0

    def _tokenize_caption(self, caption: str):
        if self.skip_tokenize:
            return [], caption
        return super()._tokenize_caption(caption)

    def __iter__(self):
        # The shipped __iter__ asserts single-init and multiplies self.metadata in place;
        # the benchmark harness iterates each loader more than once (standalone, then the
        # combined mixer), so snapshot the original metadata and reset the guard each call.
        if not hasattr(self, "_meta0"):
            self._meta0 = list(self.metadata)
        self.metadata = list(self._meta0)
        self.is_initialized = False
        return super().__iter__()

    @classmethod
    def from_jsonl(cls, jsonl_path: str, *, s3_bucket: str | None = None,
                   s3_prefix: str | None = None, **kw) -> "BenchSFTDataset":
        meta = load_sft_metadata(jsonl_path, s3_bucket=s3_bucket, s3_prefix=s3_prefix)
        return cls(meta, **kw)


__all__ = ["S3DROIDLeRobotDataset", "BenchSFTDataset", "load_sft_metadata"]
