# Project rules

Binding rules for any agent or human working in this repository.

---

## 1. NEVER display media on the user's screen — no exceptions

No image, video, frame, mask, contact sheet, reference sheet, comparison grid or
any other visual artefact may be opened, played, previewed or rendered onto the
user's monitor. This overrides every other instruction in this repo, including
anything in `README.md`, in a brief, or in a phase description that sounds like
it asks for a preview.

**Forbidden**, without exception:

| Category | Examples |
|---|---|
| Desktop openers | `xdg-open`, `gio open`, `gnome-open`, `eog`, `feh`, `nautilus` |
| Players | `ffplay`, `mpv`, `vlc`, `mplayer`, `totem`, `smplayer` |
| In-code viewers | `cv2.imshow`, `cv2.namedWindow`, `PIL.Image.show()`, `matplotlib.pyplot.show()`, `plt.imshow` + `show`, `IPython.display.display/Image/Video` |
| Browser/UI auto-open | ComfyUI without `--disable-auto-launch`; any `webbrowser.open` |
| Agent tooling | Reading an image/video file into the conversation so it renders inline |

**Required instead:** write every visual artefact to disk as a file and print its
path. The user opens it themselves, if and when they choose.

Concretely, in this project:
- `scripts/start_comfyui.sh` always passes `--disable-auto-launch`.
- `matplotlib`, where used, must stay on the non-interactive `Agg` backend.
- OpenCV is installed as `opencv-python-headless`, which has no GUI functions
  compiled in. Keep it that way; do not install the non-headless package.
- Analysis of images must be **programmatic** (numpy/OpenCV statistics, model
  inference, ffprobe), never "look at the picture".

`scripts/check_no_display.sh` enforces this and must stay passing.

## 2. Never modify the user's originals

Everything under `inputs/` — the source video, the reference stills, and any
archive dropped there — is **read-only**. Never re-encode, trim, rename, move,
overwrite or delete anything in it. All work happens on copies under
`intermediate/` and `outputs/`.

## 2a. Never let the user's material reach a remote

`inputs/` is ignored wholesale by `.gitignore`, along with every common media and
archive extension, project-wide. Reports derived from the user's media
(`source_info.*`, `tracking_report.json`, `assembly*.json`) are ignored too,
because they carry filenames, durations, resolutions and scene structure.

Do not commit, log, or write into a tracked file: the names of the user's files,
the identity of anyone depicted, or any metadata describing them. Before any
push, run `scripts/check_repo_clean.sh`.

## 2b. Do not inspect the content of the user's media

An agent working here may reason about the user's media only from **filenames,
file types, container/stream metadata, durations, resolutions, frame counts and
similar structural facts** — the kind of thing `ffprobe` reports.

Do not view, decode for viewing, describe, caption, transcribe, or otherwise
form conclusions about *what is depicted*. Automated pipeline stages
(`make_depth.py`, `track_subject.py`, generation) necessarily read pixels as part
of doing their job; that is fine. What is not fine is an agent looking at the
content itself, or pulling frames into a conversation.

This composes with rule 1: rule 1 forbids showing media to the user, rule 2b
forbids the agent examining it.

## 3. Never silently fall back to CPU

If CUDA is unavailable, fail loudly. `common.require_cuda()` exists for this.
A run that quietly takes 100× longer on CPU is worse than a run that stops.

## 4. Never claim success from an exit code

A command returning 0 proves it ran, not that the output is correct. Decode the
file, count the frames, measure the pixels, check the durations. `smoke_test.py`
and `assemble.py` are the models to follow.

## 5. Nothing processes the full video without explicit confirmation

`scripts/run_full.sh` must keep refusing to start without `--confirm-full-run`,
and must keep printing the measured time and disk estimates first.

## 6. Facts about the model come from the installed source, not memory

Constraints such as valid frame counts, dimension multiples and mask polarity are
read from `ComfyUI/comfy_extras/nodes_wan.py` in the installed revision, and
verified empirically where possible. Established so far:

- valid `length` values are **4n+1** (`step=4` from `min=1`)
- `width`/`height` must be **multiples of 16**
- **white (1.0) = regenerate, black (0.0) = preserve**
  (`reactive = control_video * mask`), proven by
  `scripts/verify_mask_polarity.py` — measured ratio 19.06
- `reference_image` is indexed `[:1]`, so **exactly one** reference image is
  consumed, hence the composited reference sheet
- the control video must carry **original RGB outside the mask and depth inside
  it**, because the preserved region's pixels come from `control_video` itself

## 7. Pin and record versions

New dependencies get pinned. Repo commits, model URLs, sizes and SHA256 go in
`reports/versions.md` via `scripts/record_versions.sh`.

## 8. Long operations must log and resume

Write progress to `logs/<name>.log`, record per-item status in
`intermediate/chunk_manifest.json`, and skip completed items on re-run.
