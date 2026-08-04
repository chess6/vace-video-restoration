#!/usr/bin/env python
"""Checks on the two rules that keep a generated result honest.

Synthetic data only: no CUDA, no ComfyUI, no video decoding, no model weights,
so this runs in a second on any checkout.

    venv/bin/python scripts/test_reference_pack.py

What it proves:

  AUTHORITY - an external photograph can only ever contribute identity
    * every garment and accessory class is excluded from what may be shown
    * every identity class is included
    * non-identity pixels really do become neutral grey, and none of the
      garment's colour survives - not even faintly, which would still be a
      garment as far as a generative model is concerned
    * identity pixels are passed through unaltered
    * the SOURCE panel is never masked: its garment pixels are bit-identical

  STALENESS - a result can never be mistaken for current
    * changing the CONTENT of a reference sheet changes the generation key, so
      an output produced from the old one stops counting as done
    * a change that lands DURING generation settles as `stale`, not `done`,
      and a stale record is always re-run

Both were regressions waiting to happen: the first because "mask the clothes"
lives in rendering code that gets edited, the second because the key used to be
read back off disk after ~16 minutes of GPU time, by which point it no longer
described the pixels that had just been produced.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import file_digest, generation_key  # noqa: E402
from make_reference_pack import (  # noqa: E402
    GARMENT, IDENTITY_ONLY, LBL, build_panel_images, identity_regions,
    mask_to_identity,
)
from run_chunks import needs_run, settled_status  # noqa: E402

GARMENT_RGB = (200, 30, 40)     # a loud colour, so any leak is unmistakable
SKIN_RGB = (170, 130, 110)
BG_RGB = (10, 200, 15)


class Failures(list):
    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.append(msg)
        return cond


def synthetic_photo(h=64, w=48):
    """A labelled stand-in for an external reference: face band over a garment
    band over background. No real image, no parser, no weights."""
    from PIL import Image
    labels = np.zeros((h, w), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = BG_RGB
    face_id = next(i for i, n in LBL.items() if n == "face")
    upper_id = next(i for i, n in LBL.items() if n == "upper")
    labels[8:24, 12:36] = face_id
    rgb[8:24, 12:36] = SKIN_RGB
    labels[28:56, 8:40] = upper_id
    rgb[28:56, 8:40] = GARMENT_RGB
    return Image.fromarray(rgb), labels


# ---------------------------------------------------------------------------
# authority: externals condition identity only
# ---------------------------------------------------------------------------

def test_class_partition(f: Failures) -> None:
    leaked = sorted(n for n in GARMENT if n in IDENTITY_ONLY)
    f.check(not leaked, f"garment class(es) {leaked} are shown to the model")

    ids = np.array(sorted(LBL), dtype=np.uint8).reshape(1, -1)
    shown = identity_regions(ids)[0]
    for i, name in LBL.items():
        want = name in IDENTITY_ONLY
        f.check(bool(shown[i]) == want,
                f"class {name!r}: shown={bool(shown[i])}, expected {want}")


def test_masking_removes_clothing(f: Failures) -> None:
    im, labels = synthetic_photo()
    src = np.asarray(im)

    out, kept = mask_to_identity(im, labels, feather=0)
    arr = np.asarray(out).astype(np.int16)
    idm = identity_regions(labels)

    f.check(np.array_equal(arr[idm], src[idm]),
            "identity pixels were altered; the face must pass through as-is")
    f.check(bool((arr[~idm] == 128).all()),
            "non-identity pixels are not neutral grey")
    f.check(abs(kept - idm.mean()) < 1e-6,
            f"reported kept fraction {kept:.4f} != actual {idm.mean():.4f}")

    # The real call feathers the edge. Feathering must soften the BOUNDARY, not
    # let garment colour bleed through the middle of the garment.
    try:
        soft = np.asarray(mask_to_identity(im, labels, feather=3)[0]).astype(np.int16)
    except ImportError:
        f.append("SKIPPED feathered check: OpenCV is not installed in this "
                 "checkout, so the blur path could not be exercised")
        return
    core = np.zeros(labels.shape, bool)
    core[34:50, 14:34] = True          # well inside the garment band
    worst = int(np.abs(soft[core] - 128).max())
    f.check(worst == 0,
            f"garment colour survives {worst}/255 into the feathered panel; "
            f"a faint jacket is still a jacket")


def test_source_panel_is_untouched(f: Failures) -> None:
    im, labels = synthetic_photo()
    panel = {"image": im, "labels": labels}
    source_crop = np.random.default_rng(0).integers(0, 256, (40, 30, 3),
                                                    dtype=np.uint8)
    imgs, keeps = build_panel_images(panel, source_crop, panel)

    f.check(len(imgs) == 3 and len(keeps) == 3,
            f"expected 3 panels with 3 kept-fractions, got {len(imgs)}/{len(keeps)}")
    f.check(np.array_equal(np.asarray(imgs[1]), source_crop),
            "the source panel was modified; its garment is the ground truth and "
            "must reach the sheet bit-identical")
    f.check(keeps[1] is None,
            "the source panel reports an identity-kept fraction, which means it "
            "went through the external masking path")
    for i in (0, 2):
        arr = np.asarray(imgs[i])
        f.check(not np.array_equal(arr, np.asarray(im)),
                f"external panel {i} reached the sheet unmasked")
        f.check(keeps[i] is not None, f"external panel {i} was not masked")

    imgs2, keeps2 = build_panel_images(panel, source_crop, None)
    f.check(len(imgs2) == 2 and keeps2[1] is None,
            "single-external pack: the source panel must still be panel 1 and "
            "must still be unmasked")


# ---------------------------------------------------------------------------
# staleness: a result can never be mistaken for current
# ---------------------------------------------------------------------------

def key_for(sheet: Path) -> str:
    """The shape of run_chunks.gen_key: content hashes, not paths."""
    return generation_key({"reference_pack": file_digest(sheet), "seed": 7})


def test_content_change_invalidates(f: Failures) -> None:
    with tempfile.TemporaryDirectory() as td:
        sheet = Path(td) / "pack.png"
        sheet.write_bytes(b"pack-with-external-clothing")
        k_old = key_for(sheet)

        chunk = {"runs": {"v": {"status": "done", "generation_key": k_old}}}
        f.check(not needs_run(chunk, "v", k_old),
                "an untouched chunk was scheduled to regenerate")

        # Same path, different bytes - exactly what rebuilding a pack does.
        sheet.write_bytes(b"pack-with-clothing-segmented-out")
        k_new = key_for(sheet)
        f.check(k_new != k_old,
                "rewriting the sheet did not change the key; a stale result "
                "would be compared as though it matched")
        f.check(needs_run(chunk, "v", k_new),
                "a chunk generated from the old pack still counts as done")


def test_midrun_change_cannot_be_recorded_current(f: Failures) -> None:
    f.check(settled_status("aaa", "aaa") == "done",
            "an undisturbed run did not settle as done")
    f.check(settled_status("aaa", "bbb") == "stale",
            "inputs changed mid-run but the result settled as done")

    # And a stale record must actually cause a re-run, whatever key is current.
    stale = {"runs": {"v": {"status": "stale", "generation_key": "aaa"}}}
    f.check(needs_run(stale, "v", "aaa"),
            "a stale record was skipped because its recorded key happens to "
            "match; the pixels behind it are still from the old inputs")
    f.check(needs_run(stale, "v", "bbb"), "a stale record was skipped")

    for st in ("pending", "failed", "running"):
        c = {"runs": {"v": {"status": st, "generation_key": "aaa"}}}
        f.check(needs_run(c, "v", "aaa"), f"status {st!r} was treated as current")
    skipped = {"runs": {"v": {"status": "skipped"}}}
    f.check(not needs_run(skipped, "v", "aaa"),
            "a shot with no subject in it was scheduled for generation")


def main() -> int:
    f = Failures()
    for t in (test_class_partition, test_masking_removes_clothing,
              test_source_panel_is_untouched, test_content_change_invalidates,
              test_midrun_change_cannot_be_recorded_current):
        t(f)
    skipped = [m for m in f if m.startswith("SKIPPED")]
    real = [m for m in f if not m.startswith("SKIPPED")]
    for m in skipped:
        print(f"  {m}")
    if real:
        print(f"FAILED: {len(real)} check(s)")
        for m in real:
            print(f"  - {m}")
        return 1
    print(f"PASSED{f' ({len(skipped)} check(s) skipped)' if skipped else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
