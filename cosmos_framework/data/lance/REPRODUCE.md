# Reproducing the LanceDB-vs-base dataloader benchmarks

Everything to recreate these numbers from scratch. All comparisons are against the **genuine
shipped base loaders**, CPU decode on both sides. Two regimes: **LOCAL** (cosmos's
pre-download-then-train workflow) and **S3** (Lance native `s3://` vs the base's only object-store
path). Plus the **memory** + **memory-scaling** benchmarks.

## 0. Hardware / OS
- Linux x86-64. A CUDA GPU is **not** required for the dataloader/memory benchmarks (decode is CPU);
  only `train_combined_e2e.py` needs a GPU.
- System `ffmpeg` (torchcodec/ffmpeg decode). FFmpeg 7 or 8.
- ~5 GB disk for the subsets + Lance tables (more for the scaling test). S3 regime needs an AWS bucket.

## 1. Python environment
Python 3.12. **torchcodec must match torch exactly**, and its `.so` needs the CUDA/NPP/ffmpeg libs on
`LD_LIBRARY_PATH` even for CPU decode. The repo's `benchmarks/lance/.venv-gpu` already does this —
`source benchmarks/lance/.venv-gpu/bin/activate` (it appends the NPP `LD_LIBRARY_PATH`). To build fresh:

```bash
python3.12 -m venv .venv-gpu && source .venv-gpu/bin/activate && python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchcodec==0.10.0+cu128
pip install nvidia-npp-cu12==12.4.1.87      # torchcodec_core*.so needs libnppicc; add its lib dir to LD_LIBRARY_PATH
printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\ntorchcodec==0.10.0+cu128\n' > /tmp/cons.txt
# data + framework deps (the VLM base imports the genuine config module, which needs the full cosmos stack):
pip install -c /tmp/cons.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
    lancedb pylance lerobot webdataset transformers peft einops datasets \
    scipy opencv-contrib-python imageio imageio-ffmpeg mediapy psutil \
    hydra-core "multi-storage-client[boto3]==0.44.0" qwen-vl-utils \
    loguru cattrs omegaconf termcolor tyro msgpack nvidia-ml-py av obstore boto3 botocore \
    pytest pytest-xdist
python -c "import torch,torchcodec,lance,lerobot; from cosmos_framework.configs.base.vlm.experiment.llava_ov_vlm import get_llava_ov_streaming; print('ok', torch.__version__)"
```
Then `export PYTHONPATH=$REPO` and (for S3) an AWS profile (default chain or `AWS_PROFILE=...`).

## 2. Datasets (public on HF)
```bash
export HF_TOKEN=...
hf download lerobot/droid_1.0.1 --repo-type dataset --local-dir <droid_raw>
hf download nvidia/BridgeData2-Subset-Synthetic-Captions --repo-type dataset --local-dir <bridge>
# LLaVA-OneVision-Data figureqa subset is streamed at run time for the base; converted for Lance below
```

## 3. Build the Lance tables + Cosmos-format subset (offline, one-time)
```bash
source benchmarks/lance/.venv-gpu/bin/activate
python tools/lance_datagen/prepare_droid_subset.py --src <droid_raw> --out <droid_out> --num-episodes 327
python tools/lance_datagen/build_composed_droid.py --root <droid_out>/success --uri <droid_lance> --gop 1 --storage plain
python tools/lance_datagen/build_vision_sft.py --jsonl <bridge>/sft_dataset_bridge/train/video_dataset_file.jsonl \
    --uri <vsft_lance> --resolution 256 --gop 1 --storage plain
python -c "from datasets import load_dataset; from cosmos_framework.data.lance.vlm_dataset import convert_llava_to_lance; \
  convert_llava_to_lance(load_dataset('lmms-lab/LLaVA-OneVision-Data', name='figureqa(cauldron,llava_format)', split='train'), '<llava_lance>')"
# (optional) bit-exact action table for the strict-parity test:
python tools/lance_datagen/build_composed_droid.py ...   # or the LanceDROIDDataset raw-bytes table
```

## 4. Equivalence — prove identical output before trusting throughput
```bash
DROID_COSMOS_ROOT=<droid_out>/success DROID_LANCE_URI=<droid_videoblob_lance> \
DROID_COMPOSED_LANCE_URI=<droid_lance> \
BRIDGE_JSONL=<bridge>/sft_dataset_bridge/train/video_dataset_file.jsonl VISION_SFT_LANCE_URI=<vsft_lance> \
HF_TOKEN=$HF_TOKEN \
  python -m pytest tests/data/lance/test_action_equivalence.py tests/data/lance/test_vision_sft_equivalence.py \
    tests/data/lance/test_vlm_equivalence.py tests/data/lance/test_batch_equivalence.py
# expect 22 passed: action bit-exact, vision-SFT token-exact, VLM byte-identical, + batch-level (composed loader)
```

## 5. Throughput benchmarks
Run `--trios base` and `--trios lance` in **separate processes** (one process hits a benign
torchcodec/lance teardown between trios). Each per-loader bench also isolates each side/mode in its own
spawn subprocess. The full matrix driver:
```bash
DATA=<data_root> S=s3://<bucket>/cosmos BUCKET=<bucket> REGION=<region> \
  bash benchmarks/lance/run_matrix.sh          # LOCAL + S3 + MIXED × {4/4/4, 18/4/18} × {base, lance}
```
Per-loader, e.g. action (LOCAL then native-S3 standin):
```bash
python benchmarks/lance/bench_action_faithful.py --root <droid_out>/success --uri <droid_lance> --num-workers 18
python benchmarks/lance/bench_action_faithful.py --root <droid_out>/success \
    --uri s3://<bucket>/.../droid_composed_plain --region <region> \
    --s3-bucket <bucket> --s3-prefix cosmos/droid327/base/success --num-workers 18
python benchmarks/lance/bench_vision_sft.py --jsonl <jsonl> --uri <vsft_lance> --mode raw --num-workers 8 18
python benchmarks/lance/bench_vlm.py --side base  --lance-uri <llava_lance> --mode raw   # run twice:
python benchmarks/lance/bench_vlm.py --side lance --lance-uri <llava_lance> --mode raw   # base, then lance; divide
```
Expected (327 eps, batch 16): combined **2.75× LOCAL / 3.65× S3** at 4/4/4, **3.32× / 4.93×** at 18/4/18;
per-loader action ~1.8×, vision-SFT ~5–8× (holds e2e), VLM raw large but ≈1× e2e. Full tables:
[`BENCHMARKS.md`](BENCHMARKS.md).

## 6. Memory + memory-scaling
```bash
# at benchmark scale (action loader, 8 workers): base is slightly leaner here (fixed Arrow runtime)
python benchmarks/lance/bench_memory.py --side base  --root <droid_out>/success --uri <droid_lance> --random
python benchmarks/lance/bench_memory.py --side lance --root <droid_out>/success --uri <droid_lance> --random
# spawn vs fork (COW): add --mp-context fork ; PSS is reported for fair COW accounting

# scaling: replicate the subset N× and re-measure — base per-worker RAM balloons (self._rows), Lance stays flat
python benchmarks/lance/build_scaled_droid.py --src-root <droid_out>/success --src-lance <droid_lance> \
    --out-root /tmp/droid_x16 --out-lance /tmp/lance_x16 --table droid_composed --n 16
ln -sfn <droid_out>/success/videos /tmp/droid_x16/videos
for s in base lance; do python benchmarks/lance/bench_memory.py --side $s --root /tmp/droid_x16 --uri /tmp/lance_x16 --random; done
```
Expected: per-worker PSS at 16× (1.54M frames) base ~2.6 GB vs Lance ~0.86 GB (crossover ≈ 4×). Details:
[`BENCHMARKS.md`](BENCHMARKS.md) §3.

## 7. E2E training (optional, needs a GPU)
```bash
python benchmarks/lance/train_combined_e2e.py --trio {base,lance} --regime {local,s3,mixed} --layers L \
   --action-workers 18 --vlm-workers 4 --vsft-workers 18
```
Sweeping `--layers` traces the data-bound → compute-bound crossover (at realistic model size single-GPU
training is compute-bound → base ≈ lance wall-clock; the loader win surfaces data-bound / fast-GPU / remote).

## 8. Gotchas
- **Same decode device both sides** — always CPU. The base can't use GPU.
- **Separate process per trio** for the combined bench; per-loader benches self-isolate each measurement.
- **The combined number is bottleneck-gated** (≈ slowest loader); report the per-loader breakdown with it.
- **Genuine bases, no FUSE**: action S3 uses the `S3DROIDLeRobotDataset` standin (download then genuine
  decode), vision-SFT uses the genuine `SFTDataset` per-sample boto3, VLM uses `get_llava_ov_streaming`.
  These are fairer (and lower) than the old FUSE-based S3 numbers.
- **VLM raw ratio is access-only** (≈1× e2e). Don't quote it as training speedup.
