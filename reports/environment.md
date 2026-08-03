# Environment Report

Generated: 2026-08-03T15:32:47-07:00
Project root: `/home/thomas/Dev/upscaler/vace-video-restoration`

## Summary table

| Item | Value | Status |
|---|---|---|
| Ubuntu | `Ubuntu 24.04.4 LTS` | OK |
| Kernel | `6.8.0-136-generic` | OK |
| CPU | `AMD Ryzen 9 5900X 12-Core Processor` | OK |
| RAM total / available | `15 GiB / 3 GiB` | OK |
| Swap | `15 GiB` | OK |
| GPU | `NVIDIA GeForce RTX 3060` | OK |
| VRAM total / free | `12288 MiB / 3991 MiB` | OK |
| NVIDIA driver | `580.173.02` | OK |
| Compute capability | `8.6` | OK |
| Free disk at project | `590 GiB` | OK |
| git | `unknown option: -version` | OK |
| ffmpeg | `ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FF` | OK |
| ffprobe | `ffprobe version 6.1.1-3ubuntu5 Copyright (c) 2007-2023 the F` | OK |
| curl | `curl: (2) no URL specified` | OK |
| venv python | `Python 3.12.3` | OK |
| ComfyUI commit | `e377e263049f9338b4d12a3dd417b36ae62948ff` | OK |
| ComfyUI tag | `v0.30.0` | OK |
| PyTorch | `2.9.1+cu128 (bundled CUDA 12.8)` | OK |
| torch.cuda.is_available() | `True -> NVIDIA GeForce RTX 3060` | OK |

**Checks passed: 19 — failed: 0**

## Key findings that changed the plan

1. **The GPU has 12 GB of VRAM, not 8 GB.** `nvidia-smi` reports an RTX 3060 with
   12288 MiB. The brief assumed ~8 GB. 12 GB comfortably fits VACE 1.3B at 480p and
   makes the FP8 text encoder a convenience rather than a hard requirement. The
   14B model is still **not** attempted locally, per instruction.

2. **System RAM is the real bottleneck, not VRAM.** 15 GiB total, with a large part
   already used by the desktop session (Firefox, Cursor, GNOME). The FP8 UMT5 text
   encoder alone is ~6.3 GB and is staged through system RAM.

   Consequences for the launcher flags:
   - `--cache-none` **is** used: it stops ComfyUI caching every node's output
     between runs, which is the single biggest RAM saver here.
   - `--disable-smart-memory` is deliberately **not** used. Its actual behaviour is
     to aggressively offload weights from VRAM back into system RAM, which is
     exactly the wrong trade on this machine: VRAM is plentiful (12 GB) and RAM is
     scarce. Leaving smart memory enabled keeps the weights resident in VRAM.
   - `--reserve-vram 1.0` leaves ~1 GB for Xorg/Firefox, which were measured
     holding ~1.1 GB at inspection time.

   Closing Firefox and Cursor before a long run frees several GiB of RAM.

3. **No system CUDA toolkit is installed and none is needed.** PyTorch ships its own
   CUDA 12.8 runtime. Driver 580.x supports it. The system driver was left untouched.

4. **Python 3.10 on PATH is Miniconda's.** The project deliberately uses the *system*
   `python3.12` to build `venv/`, so the conda base environment is never modified.

## Disk space budget

| Item | Size |
|---|---|
| PyTorch + CUDA runtime wheels | ~8.5 GB |
| ComfyUI + Python deps | ~1.5 GB |
| `wan2.1_vace_1.3B_fp16.safetensors` | ~3.2 GB |
| `wan_2.1_vae.safetensors` | ~0.25 GB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.7 GB |
| Depth Anything V2 (ViT-L) | ~1.3 GB |
| SAM 2.1 (hiera-large) | ~0.9 GB |
| Detector + ReID + face embeddings | ~1.5 GB |
| **Install subtotal** | **~24 GB** |
| Intermediates for 30 min @ 16 fps 832x480 (depth + masks + chunks) | ~45-70 GB |
| Restored 480p master + deliverables | ~15-25 GB |
| **Peak total** | **~85-120 GB** |

Free space measured above must exceed the peak total before a full run.

## Missing system packages

None. No `sudo` is required for this project.
