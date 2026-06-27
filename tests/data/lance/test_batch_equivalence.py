# SPDX-License-Identifier: OpenMDW-1.1
"""Batch-level equivalence: the BENCHMARKED Lance loaders must produce the same batch
as the base loader through the ``__getitems__`` hot path (what the DataLoader actually
calls), not just single ``__getitem__`` samples.

This complements the per-sample tests and, unlike them, covers the *fast composed* action
loader (``LanceDROIDComposedDataset``) that the benchmarks use. The equivalence guarantee
differs by loader and is asserted exactly as such:

  * action labels/poses/captions, vision-SFT token-ids, VLM records → **exact**;
  * fast-path video (composed action, vision-SFT) → within **H.264 re-encode tolerance**
    (the stored clip is the base's exact transform re-encoded once); the bit-exact
    raw-bytes ``LanceDROIDDataset`` variant is also checked to be **pixel-identical**.

Set the fixture env vars to run (skipped otherwise):
  DROID_COSMOS_ROOT, DROID_COMPOSED_LANCE_URI (+ optional DROID_LANCE_URI for raw bytes),
  BRIDGE_JSONL, VISION_SFT_LANCE_URI, and HF_TOKEN for the VLM check.
"""
from __future__ import annotations

import os

import pytest
import torch

_BATCH = [0, 1, 123, 5000, 17000, 26000]
_VMAD = 0.05  # mean |Δ|/255 tolerance for re-encoded video


def _video_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item() / 255.0


# ── action (joint_pos): composed loader (benchmarked) + raw-bytes loader ──
_AROOT = os.environ.get("DROID_COSMOS_ROOT")
_ACOMP = os.environ.get("DROID_COMPOSED_LANCE_URI")
_ARAW = os.environ.get("DROID_LANCE_URI")
_AKW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


@pytest.mark.skipif(not (_AROOT and _ACOMP and os.path.isdir(_AROOT)),
                    reason="set DROID_COSMOS_ROOT + DROID_COMPOSED_LANCE_URI")
def test_action_composed_batch_matches_base():
    from cosmos_framework.data.lance import LanceDROIDComposedDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    base = DROIDLeRobotDataset(root=_AROOT, **_AKW)
    lance = LanceDROIDComposedDataset(root=_AROOT, lance_uri=_ACOMP, decode_device="cpu", **_AKW)
    idxs = [i for i in _BATCH if i < len(base)]
    batch = lance.__getitems__(idxs)  # the DataLoader hot path
    assert len(batch) == len(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert b.keys() == l.keys()
        assert torch.equal(b["action"], l["action"]), f"action differs at {i}"
        assert int(b["idle_frames"]) == int(l["idle_frames"])
        assert b["ai_caption"] == l["ai_caption"]
        assert b["video"].shape == l["video"].shape
        mad = _video_mad(b["video"], l["video"])
        assert mad < _VMAD, f"video mean|Δ|/255={mad:.4f} at {i} exceeds {_VMAD}"


@pytest.mark.skipif(not (_AROOT and _ARAW and os.path.isdir(_AROOT)),
                    reason="set DROID_COSMOS_ROOT + DROID_LANCE_URI (raw-bytes table)")
def test_action_rawbytes_batch_pixel_identical():
    from cosmos_framework.data.lance import LanceDROIDDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    base = DROIDLeRobotDataset(root=_AROOT, **_AKW)
    lance = LanceDROIDDataset(root=_AROOT, lance_uri=_ARAW, decode_device="cpu", **_AKW)
    idxs = [i for i in _BATCH if i < len(base)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["video"], l["video"]), f"video not pixel-identical at {i}"
        assert torch.equal(b["action"], l["action"])


# ── vision-SFT: tokens exact, video within tolerance ──
_VJSONL = os.environ.get("BRIDGE_JSONL")
_VURI = os.environ.get("VISION_SFT_LANCE_URI")
_VKW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


@pytest.mark.skipif(not (_VJSONL and _VURI and os.path.isfile(_VJSONL)),
                    reason="set BRIDGE_JSONL + VISION_SFT_LANCE_URI")
def test_vision_sft_batch_matches_base():
    from cosmos_framework.data.lance import LanceVisionSFTDataset
    from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import LocalSFTDataset

    base = LocalSFTDataset(_VJSONL, **_VKW)
    lance = LanceVisionSFTDataset(_VURI, table="vision_sft", decode_device="cpu", **_VKW)
    idxs = [i for i in [0, 1, 17, 50, 123] if i < len(base)]
    batch = lance.__getitems__(idxs)
    assert len(batch) == len(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["text_token_ids"], l["text_token_ids"]), f"token ids differ at {i}"
        assert b["ai_caption"] == l["ai_caption"]
        assert b["video"].shape == l["video"].shape
        assert _video_mad(b["video"], l["video"]) < _VMAD


# ── VLM: records byte-identical ──
@pytest.mark.skipif(not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
                    reason="set HF_TOKEN to stream LLaVA-OneVision base records")
def test_vlm_batch_matches_base():
    import tempfile

    from datasets import load_dataset

    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset, convert_llava_to_lance

    subset = os.environ.get("LLAVA_SUBSET", "figureqa(cauldron,llava_format)")
    stream = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split="train", streaming=True)
    stream = stream.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)
    base = []
    for rec in stream:
        base.append(rec)
        if len(base) >= 16:
            break
    tmp = tempfile.mkdtemp()
    convert_llava_to_lance(iter(base), tmp, table_name="llava")
    lance = LanceVLMDataset(tmp, table_name="llava")
    batch = lance.__getitems__(list(range(len(base))))
    for i, l in enumerate(batch):
        assert l["conversations"] == (base[i].get("conversations") or []), f"conversations differ at {i}"
        import io

        img = base[i].get("image")
        raw = img.get("bytes") if isinstance(img, dict) else (
            (lambda buf: (img.save(buf, format=img.format or "PNG"), buf.getvalue())[1])(io.BytesIO()))
        assert l["image"]["bytes"] == raw, f"image bytes differ at {i}"
