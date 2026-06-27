# SPDX-License-Identifier: OpenMDW-1.1
"""The LanceDB vision-SFT loader must match the GENUINE shipped ``SFTDataset``.

The base reference here is the real ``SFTDataset.process_one_sample`` (not a mirror): we
build the shipped dataset and call it on the same single-window metadata each Lance row was
built from, which is deterministic for this config (``frame_selection_mode="first"``,
``cfg_dropout_rate=0``, one window per sample). Caption/token-ids are **exact**; video is
near-identical (one offline H.264 re-encode of the base's exact resize).

    BRIDGE_JSONL=.../sft_dataset_bridge/train/video_dataset_file.jsonl \
    VISION_SFT_LANCE_URI=.../lance/vision_sft \
    pytest tests/data/lance/test_vision_sft_equivalence.py
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

JSONL = os.environ.get("BRIDGE_JSONL")
URI = os.environ.get("VISION_SFT_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (JSONL and URI and os.path.isfile(JSONL)),
    reason="set BRIDGE_JSONL and VISION_SFT_LANCE_URI to the prepared fixtures",
)

_KW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")
_VMAD = 0.02  # mean |Δ|/255 tolerance for the one offline re-encode (measured ~1.3%)


def _genuine_sftdataset_and_metas(jsonl: str):
    """Build the shipped SFTDataset + the per-window metadata list in the SAME order the Lance
    table was built (build_vision_sft iterates jsonl records × windows), so Lance row i is the
    genuine ``process_one_sample(metas[i])``."""
    from transformers import AutoTokenizer

    from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
    from cosmos_framework.data.vfm.local_datasets.sft_dataset import SFTDataset

    base_dir = os.path.dirname(os.path.abspath(jsonl))
    metas = []
    with open(jsonl) as f:
        for line in f:
            rec = json.loads(line)
            vp = rec["vision_path"]
            vp = vp if ("://" in vp or vp.startswith("/")) else os.path.join(base_dir, vp)
            for wi, w in enumerate(rec["t2w_windows"]):
                metas.append({
                    "uuid": f"{rec['uuid']}_w{wi}", "vision_path": vp,
                    "width": rec["width"], "height": rec["height"],
                    "aspect_ratio": get_aspect_ratio(rec["width"], rec["height"]),
                    "t2w_windows": [w],
                })
    tok_cfg = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"))
    ds = SFTDataset(metadata=metas, num_video_frames=16, resolution="256", s3_credentials={},
                    frame_selection_mode="first", temporal_interval_mode="entire_chunk",
                    tokenizer_config=tok_cfg, cfg_dropout_rate=0.0)
    ds.s3_client = None  # local vision_path -> download_from_s3 falls back to Path.read_bytes
    return ds, metas


@pytest.fixture(scope="module")
def loaders():
    from cosmos_framework.data.lance import LanceVisionSFTDataset

    base, metas = _genuine_sftdataset_and_metas(JSONL)
    lance = LanceVisionSFTDataset(URI, table="vision_sft", decode_device="cpu", **_KW)
    return base, metas, lance


def test_same_length(loaders):
    base, metas, lance = loaders
    assert len(metas) == len(lance)


@pytest.mark.parametrize("idx", [0, 1, 17, 50, 123, 199])
def test_sample_equivalent_vs_genuine(loaders, idx):
    base, metas, lance = loaders
    if idx >= len(lance):
        pytest.skip("index beyond dataset")
    ref = base.process_one_sample(metas[idx])  # the GENUINE shipped per-sample processing
    assert ref is not None, "genuine SFTDataset returned None"
    l = lance[idx]
    assert l["__key__"] == metas[idx]["uuid"], "row order mismatch (lance vs jsonl)"
    # caption + token ids EXACT vs the shipped loader
    assert ref["ai_caption"] == l["ai_caption"], "caption differs from genuine SFTDataset"
    assert torch.equal(ref["text_token_ids"], l["text_token_ids"]), "token ids differ from genuine"
    # video shape exact; pixels within one offline H.264 re-encode
    assert ref["video"].shape == l["video"].shape, f"{ref['video'].shape} != {l['video'].shape}"
    assert ref["num_frames"] == l["num_frames"]
    mad = (ref["video"].float() - l["video"].float()).abs().mean().item() / 255.0
    assert mad < _VMAD, f"mean|Δ|/255 = {mad:.4f} too large vs genuine SFTDataset"
