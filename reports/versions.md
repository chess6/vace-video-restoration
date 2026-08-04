# Pinned versions and model provenance

Generated: 2026-08-03T16:23:09-07:00

## Repositories

| Component | Source | Commit / version |
|---|---|---|
| ComfyUI | https://github.com/comfyanonymous/ComfyUI | `e377e263049f9338b4d12a3dd417b36ae62948ff` (v0.30.0) |
| SAM 2 | https://github.com/facebookresearch/sam2 | pinned `2b90b9f5ceec907a1c18123530e92e794ad901a4` (v1.0) |

**Custom ComfyUI nodes installed: none.**

The baseline workflow uses only native nodes (`WanVaceToVideo`,
`LoadVideo`, `GetVideoComponents`, `ImageCompositeMasked`, `ImageToMask`,
`FeatherMask`, `CreateVideo`, `SaveVideo`, …), so there are no third-party
nodes to pin or to break on a ComfyUI update. Depth estimation, detection
and SAM 2 tracking run as separate processes outside ComfyUI, which also
keeps them from competing with VACE for VRAM.

## Python environment

```
torch                      2.9.1+cu128
torchvision                0.24.1+cu128
torchaudio                 2.9.1+cu128
transformers               4.56.1
huggingface-hub            0.36.2
opencv-python-headless     5.0.0.93
numpy                      2.2.6
scenedetect                0.7.1
insightface                1.0.1
onnxruntime-gpu            1.28.0
timm                       1.0.28
SAM-2                      1.0
safetensors                0.8.0
Pillow                     12.2.0
scipy                      1.18.0
```

## Model files

| File | Bytes | SHA256 | Source |
|---|---|---|---|
| `wan2.1_vace_1.3B_fp16.safetensors` | 4309519800 | `640ccc0577e6a5d4…` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6735906897 | `c3355d30191f1f06…` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors |
| `wan_2.1_vae.safetensors` | 253815318 | `2fc39d31359a4b0a…` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors |

Full checksums are re-verified on every run of `scripts/download_models.sh`.

## Background restoration (SeedVR2)

Downloaded by `scripts/download_seedvr2.sh`, checksum-verified on every run.
Support is **native to the pinned ComfyUI revision**
(`comfy_extras/nodes_seedvr.py`, `comfy/ldm/seedvr/`), so no custom node pack is
installed and the node semantics move only when the pinned commit moves.

| File | Purpose | Size | SHA256 |
|---|---|---|---|
| `diffusion_models/seedvr2_3b_fp8_e4m3fn.safetensors` | full-frame restoration, 3B in fp8 | 3.39 GB | `a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914` |
| `vae/seedvr2_ema_vae_fp16.safetensors` | SeedVR2 VAE | 0.50 GB | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |

Repository: `Comfy-Org/SeedVR2`.

**Deliberately not downloaded: SeedVR2 7B**, in either precision. It does not fit
12 GB at video length, and the project refuses to fall back to CPU (rule 3).

Constraints read from the installed nodes rather than from documentation:

- `SeedVR2Preprocess` pads to a multiple of **16** in both axes and to **4n+1**
  frames — the same two constraints VACE has
- `SeedVR2TemporalChunk.frames_per_chunk` must be **4n+1**; `temporal_overlap` is
  counted in **latent** frames and is clamped to `(frames_per_chunk-1)/4`
- `chunking_mode` is a `COMFY_DYNAMICCOMBO_V3`: the API prompt carries the option
  key under the input name and each nested input under `<parent>.<child>`
- `SeedVR2PostProcessing` colour-matches the result to the pre-upscale frames
  (`lab` / `wavelet` / `adain` / `none`)

## Auxiliary models (Hugging Face, cached in `hf_cache/`)

Every one of these is loaded at an explicit `revision=`, so `from_pretrained`
cannot follow a moved branch and change the output of the same code later. The
constants live next to the loaders (`scripts/track_subject.py`,
`scripts/make_depth.py`).

| Model | Purpose | Pinned revision |
|---|---|---|
| `facebook/sam2.1-hiera-large` | full-figure mask tracking | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| `IDEA-Research/grounding-dino-base` | open-vocabulary person detection | `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` |
| `openai/clip-vit-large-patch14` | appearance / clothing ReID embedding | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| `depth-anything/Depth-Anything-V2-Large-hf` | depth structural control | `7581137eff8d4e94f6e796d3baea0e9fa79b22d2` |
| `insightface buffalo_l` (ArcFace) | face identity embedding | bundled model zip, checksummed by `download_aux_models.sh` |

## Reproducing this environment

`scripts/bootstrap.sh` recreates everything from a fresh clone: the pinned
ComfyUI commit, a Python 3.12 venv installed from `requirements.lock.txt`
(exact versions for all 132 packages, torch included), and the checksummed
weights. `scripts/bootstrap.sh --check` reports what is missing without
changing anything.

## Deliberately NOT installed

- Wan2.1-VACE-**14B** weights — will not fit 12 GB; see `configs/cloud_14b.yaml`
- extra T2V / I2V checkpoints
- acceleration LoRAs (CausVid etc.) — excluded from the baseline by design
- GFPGAN / CodeFormer — face-only restoration was explicitly ruled out
- Real-ESRGAN — optional *post*-restoration resize only, and only if it
  beats Lanczos in `scripts/compare_upscalers.py`
