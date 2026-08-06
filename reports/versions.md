# Pinned versions and model provenance

Generated: 2026-08-05T11:09:04+00:00

## Repositories

| Component | Source | Commit / version |
|---|---|---|
| ComfyUI | https://github.com/comfyanonymous/ComfyUI | `e377e263049f9338b4d12a3dd417b36ae62948ff` (untagged) |
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
numpy                      2.5.1
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

## Auxiliary models (Hugging Anchor, cached in `hf_cache/`)

| Model | Purpose |
|---|---|
| `facebook/sam2.1-hiera-large` | full-subject mask tracking |
| `IDEA-Research/grounding-dino-base` | open-vocabulary candidate detection |
| `openai/clip-vit-large-patch14` | appearance / attributes ReID embedding |
| `depth-anything/Depth-Anything-V2-Large-hf` | depth structural control |
| `insightface buffalo_l` (ArcFace) | anchor match embedding |

## Subject-LoRA training (musubi-tuner)

| Component | Source | Commit / version |
|---|---|---|
| musubi-tuner | https://github.com/kohya-ss/musubi-tuner | `30c658c4f4b0bf05038b3346eff9670259b10fc7` (v0.3.4) |

Trainer weights are a SEPARATE set from the pipeline's, and deliberately so:
musubi has no VACE task, and rejects the fp8_scaled text encoder ComfyUI uses.
The VAE is shared, not duplicated.

| File | Bytes | SHA256 | Source |
|---|---|---|---|
| `wan2.1_t2v_1.3B_bf16.safetensors` | 2838104528 | `6f999b0d6cb9a72b…` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11361920418 | `7cace0da2b446bbb…` | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth |

## Deliberately NOT installed

- Wan2.1-VACE-**14B** weights — will not fit 12 GB; see `configs/cloud_14b.yaml`
- extra T2V / I2V checkpoints
- acceleration LoRAs (CausVid etc.) — excluded from the baseline by design
- GFPGAN / CodeFormer — anchor-only restoration was explicitly ruled out
- Real-ESRGAN — optional *post*-restoration resize only, and only if it
  beats Lanczos in `scripts/compare_upscalers.py`
