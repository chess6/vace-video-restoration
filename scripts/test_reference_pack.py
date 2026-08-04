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
from common import composite_key, file_digest, generation_key  # noqa: E402
from make_reference_pack import (  # noqa: E402
    GARMENT, IDENTITY_ONLY, LBL, build_panel_images, consensus_identity,
    face_yaw, identity_regions, mask_to_identity, score_identity,
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
    """The shape of run_chunks.vace_key: content hashes, not paths."""
    return generation_key({"reference_pack": file_digest(sheet), "seed": 7})


def test_content_change_invalidates(f: Failures) -> None:
    with tempfile.TemporaryDirectory() as td:
        sheet = Path(td) / "pack.png"
        sheet.write_bytes(b"pack-with-external-clothing")
        k_old = key_for(sheet)

        chunk = {"runs": {"v": {"status": "done", "vace_key": k_old}}}
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

        # A record carrying only the PRE-SPLIT combined key is not current by
        # itself. It goes through the explicit migration in run_chunks, which
        # checks that no generation input has been touched since; it must never
        # pass silently here.
        legacy = {"runs": {"v": {"status": "done", "generation_key": k_new}}}
        f.check(needs_run(legacy, "v", k_new),
                "a pre-split record was accepted as current without the "
                "migration ever verifying its inputs")


def test_composite_key_is_independent_of_generation(f: Failures) -> None:
    """The point of splitting the keys: a cheap compositing change must not send
    a finished 18-minute generation back to the GPU."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for n in ("out.mp4", "plate.mkv", "mask.mkv", "occ.mkv"):
            (d / n).write_bytes(n.encode())
        settings = {"band_px": 3, "occluder_band_px": 2}
        args = (d / "out.mp4", d / "plate.mkv", d / "mask.mkv", d / "occ.mkv")

        k1 = composite_key(*args, settings)
        k2 = composite_key(*args, {**settings, "occluder_band_px": 4})
        f.check(k1 != k2, "changing a compositing setting left the composite "
                          "key unchanged, so the composite would go stale "
                          "without anyone noticing")

        # ...and the same change must be invisible to the VACE key, which is
        # built from sampler inputs only.
        vk = generation_key({"reference_pack": file_digest(d / "out.mp4"),
                             "seed": 7})
        f.check(not needs_run({"runs": {"v": {"status": "done", "vace_key": vk}}},
                              "v", vk),
                "a compositing change forced a regeneration")

        # Regenerating the subject MUST invalidate the composite built on it.
        (d / "out.mp4").write_bytes(b"a different generation")
        f.check(composite_key(*args, settings) != k1,
                "the composite key ignored new generated pixels")


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


# ---------------------------------------------------------------------------
# identity: consensus, not a single maximum
# ---------------------------------------------------------------------------

def person(rng, base, jitter=0.12, size=(800, 600), frac=0.05, yaw=0.0):
    """A synthetic reference: a face embedding near `base`, plus the metadata the
    selector reads. Background and outfit are irrelevant here by construction -
    which is the point, since identity must not depend on them."""
    v = base + rng.normal(0, jitter, base.shape)
    v = v / np.linalg.norm(v)
    m = {"face": v.astype(np.float32), "face_frac": frac, "size": list(size),
         "yaw": yaw, "cluster": 0}
    m["file"] = type("F", (), {"name": f"ref{rng.integers(1 << 30)}.jpg"})()
    return m


def test_consensus_beats_duplicate_outliers(f: Failures) -> None:
    """Three copies of the WRONG person, which is what defeats a maximum.

    Taking the single largest similarity asks only whether SOME other photograph
    agrees. Duplicates of an impostor agree with each other perfectly, so the
    impostor scored 1.000 and could be drawn as the identity panel.
    """
    rng = np.random.default_rng(7)
    right = np.array([1.0] + [0.0] * 15)
    wrong = np.array([0.0, 1.0] + [0.0] * 14)

    real = [person(rng, right) for _ in range(5)]
    impostor = person(rng, wrong, jitter=0.0)
    dupes = [impostor] + [dict(impostor, file=impostor["file"]) for _ in range(2)]
    items = real + dupes

    info = consensus_identity(items)
    score_identity(items)

    f.check(info["duplicates"] >= 2,
            f"near-duplicates were not collapsed ({info['duplicates']} found); "
            f"repeated copies can still vote for each other")
    for m in real:
        f.check(m["identity"]["verified"],
                "a genuine reference was excluded from the consensus group")
    for m in dupes:
        f.check(not m["identity"]["verified"],
                "an impostor backed only by copies of itself was verified; this "
                "is exactly what taking the maximum similarity got wrong")
        f.check(m["identity"]["score"] < min(r["identity"]["score"] for r in real),
                "an impostor outscored every genuine reference")


def test_consensus_survives_background_and_outfit_changes(f: Failures) -> None:
    """Same person, photographed in different places wearing different things.

    Identity must not weaken because the scene changed - that was the flaw in
    reading full-image similarity as identity or as viewpoint.
    """
    rng = np.random.default_rng(11)
    base = np.array([0.0, 0.0, 1.0] + [0.0] * 13)
    items = [person(rng, base, jitter=0.20) for _ in range(6)]
    for i, m in enumerate(items):          # wildly different scenes and clothes
        m["clip"] = np.eye(16)[i].astype(np.float32)
        m["clip_identity"] = (base + rng.normal(0, 0.05, 16))
        m["clip_identity"] /= np.linalg.norm(m["clip_identity"])

    consensus_identity(items)
    score_identity(items)
    f.check(all(m["identity"]["verified"] for m in items),
            "changing background and outfit broke identity consensus, which "
            "should depend on the face alone")


def test_yaw_is_a_head_measurement(f: Failures) -> None:
    """Landmarks, not scene similarity."""
    f.check(face_yaw(None) is None and face_yaw([(0, 0)]) is None,
            "missing landmarks must yield no yaw rather than a made-up one")

    head_on = [(40, 50), (60, 50), (50, 62), (44, 72), (56, 72)]
    turned = [(40, 50), (60, 50), (57, 62), (44, 72), (56, 72)]
    other = [(40, 50), (60, 50), (43, 62), (44, 72), (56, 72)]
    yf, yt, yo = face_yaw(head_on), face_yaw(turned), face_yaw(other)
    f.check(abs(yf) < 0.05, f"a head-on view reported yaw {yf:+.3f}")
    f.check(yt > 0.2 and yo < -0.2,
            f"turned faces did not separate by sign ({yt:+.3f}, {yo:+.3f})")

    # Scaling and translating the whole face must not change the yaw: it is an
    # orientation, not a position or a size.
    big = [(2 * x + 300, 2 * y + 100) for x, y in turned]
    f.check(abs(face_yaw(big) - yt) < 1e-6,
            "yaw changed when the face was moved and scaled")


def main() -> int:
    f = Failures()
    for t in (test_class_partition, test_masking_removes_clothing,
              test_source_panel_is_untouched, test_content_change_invalidates,
              test_composite_key_is_independent_of_generation,
              test_midrun_change_cannot_be_recorded_current,
              test_consensus_beats_duplicate_outliers,
              test_consensus_survives_background_and_outfit_changes,
              test_yaw_is_a_head_measurement):
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
