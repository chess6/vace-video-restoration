#!/usr/bin/env python
"""Checks on the two rules that keep a generated result honest.

Synthetic data only: no CUDA, no ComfyUI, no video decoding, no model weights,
so this runs in a second on any checkout.

    venv/bin/python scripts/test_reference_pack.py

What it proves:

  AUTHORITY - an external reference can only ever contribute match
    * every attribute and accessory class is excluded from what may be shown
    * every match class is included
    * non-match pixels really do become neutral grey, and none of the
      attribute's colour survives - not even faintly, which would still be an
      attribute as far as a generative model is concerned
    * match pixels are passed through unaltered
    * the SOURCE panel is never masked: its attribute pixels are bit-identical

  STALENESS - a result can never be mistaken for current
    * changing the CONTENT of a reference sheet changes the generation key, so
      an output produced from the old one stops counting as done
    * a change that lands DURING generation settles as `stale`, not `done`,
      and a stale record is always re-run

Both were regressions waiting to happen: the first because "mask the attributes"
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
    anchor_orientation, build_panel_images,
    consensus_match, match_regions, mask_to_match, score_match,
)
from run_chunks import needs_run, settled_status  # noqa: E402

# Resolved lazily now: these used to be module constants in
# make_reference_pack, which made importing it require the untracked
# binding. Accessors keep the fail-loud behaviour and move it to first use.
from make_reference_pack import group as _group, label_map as _label_map  # noqa: E402
from test_fixtures import synthetic_bindings  # noqa: E402


def ATTRIBUTE(): return _group("attribute")
def MATCH_ONLY(): return _group("match_only")
def LBL(): return _label_map()


ATTRIBUTE_RGB = (200, 30, 40)     # a loud colour, so any leak is unmistakable
EXPOSED_RGB = (170, 130, 110)
BG_RGB = (10, 200, 15)


class Failures(list):
    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.append(msg)
        return cond


def synthetic_reference(h=64, w=48):
    """A labelled stand-in for an external reference: anchor band over an attribute
    band over background. No real image, no parser, no weights."""
    from PIL import Image
    labels = np.zeros((h, w), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = BG_RGB
    # Derived from the GROUPS, never from label names spelled out here. Naming
    # them put category words in a tracked file, and tied the test to one map.
    anchor_id = next(i for i, n in LBL().items() if n in _group("anchor_region"))
    upper_id = next(i for i, n in LBL().items() if n in _group("attribute"))
    labels[8:24, 12:36] = anchor_id
    rgb[8:24, 12:36] = EXPOSED_RGB
    labels[28:56, 8:40] = upper_id
    rgb[28:56, 8:40] = ATTRIBUTE_RGB
    return Image.fromarray(rgb), labels


# ---------------------------------------------------------------------------
# authority: externals condition match only
# ---------------------------------------------------------------------------

def test_class_partition(f: Failures) -> None:
    leaked = sorted(n for n in ATTRIBUTE() if n in MATCH_ONLY())
    f.check(not leaked, f"attribute class(es) {leaked} are shown to the model")

    ids = np.array(sorted(LBL()), dtype=np.uint8).reshape(1, -1)
    shown = match_regions(ids)[0]
    for i, name in LBL().items():
        want = name in MATCH_ONLY()
        f.check(bool(shown[i]) == want,
                f"class {name!r}: shown={bool(shown[i])}, expected {want}")


def test_masking_removes_attributes(f: Failures) -> None:
    im, labels = synthetic_reference()
    src = np.asarray(im)

    out, kept = mask_to_match(im, labels, feather=0)
    arr = np.asarray(out).astype(np.int16)
    idm = match_regions(labels)

    f.check(np.array_equal(arr[idm], src[idm]),
            "match pixels were altered; the anchor must pass through as-is")
    f.check(bool((arr[~idm] == 128).all()),
            "non-match pixels are not neutral grey")
    f.check(abs(kept - idm.mean()) < 1e-6,
            f"reported kept fraction {kept:.4f} != actual {idm.mean():.4f}")

    # The real call feathers the edge. Feathering must soften the BOUNDARY, not
    # let attribute colour bleed through the middle of the attribute.
    try:
        soft = np.asarray(mask_to_match(im, labels, feather=3)[0]).astype(np.int16)
    except ImportError:
        f.append("SKIPPED feathered check: OpenCV is not installed in this "
                 "checkout, so the blur path could not be exercised")
        return
    # NOTHING outside the match region may keep any of its original value,
    # not merely the middle of the attribute. A blur that ramps outwards leaves a
    # ring of real pixels around the anchor region - measured at 83/255 on a real
    # reference - and what lies immediately around it is attribute-bearing.
    outside = ~match_regions(labels)
    worst = int(np.abs(soft[outside] - 128).max())
    f.check(worst == 0,
            f"{worst}/255 of the original survives OUTSIDE the match region "
            f"after feathering; the ramp must lie inside it, never spread out "
            f"of it")
    f.check(not np.array_equal(soft[match_regions(labels)],
                               np.full(int(match_regions(labels).sum()) * 3,
                                       128).reshape(-1, 3)),
            "the feathered panel blanked the match region too")


def test_source_panel_is_untouched(f: Failures) -> None:
    im, labels = synthetic_reference()
    panel = {"image": im, "labels": labels}
    source_crop = np.random.default_rng(0).integers(0, 256, (40, 30, 3),
                                                    dtype=np.uint8)
    # build_panel_images feathers by default, and feathering needs OpenCV. The
    # skip is the same idiom the feathered check above already uses: an absent
    # third-party package is an environment fact, and turning it into a failure
    # trains the reader to ignore a red line. The BINDING-dependent half of this
    # test has already run by the time we get here.
    try:
        imgs, keeps = build_panel_images(panel, source_crop, panel)
    except ImportError:
        f.append("SKIPPED source-panel check: OpenCV is not installed in this "
                 "checkout, so the panel-building path could not be exercised")
        return

    f.check(len(imgs) == 3 and len(keeps) == 3,
            f"expected 3 panels with 3 kept-fractions, got {len(imgs)}/{len(keeps)}")
    f.check(np.array_equal(np.asarray(imgs[1]), source_crop),
            "the source panel was modified; its attribute is the ground truth and "
            "must reach the sheet bit-identical")
    f.check(keeps[1] is None,
            "the source panel reports an match-kept fraction, which means it "
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
        sheet.write_bytes(b"pack-with-external-attributes")
        k_old = key_for(sheet)

        chunk = {"runs": {"v": {"status": "done", "vace_key": k_old}}}
        f.check(not needs_run(chunk, "v", k_old),
                "an untouched chunk was scheduled to regenerate")

        # Same path, different bytes - exactly what rebuilding a pack does.
        sheet.write_bytes(b"pack-with-attributes-segmented-out")
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
# match: consensus, not a single maximum
# ---------------------------------------------------------------------------

def candidate(rng, base, jitter=0.12, size=(800, 600), frac=0.05, turn=0.0):
    """A synthetic reference: an anchor embedding near `base`, plus the metadata the
    selector reads. Background and appearance are irrelevant here by construction -
    which is the point, since match must not depend on them."""
    v = base + rng.normal(0, jitter, base.shape)
    v = v / np.linalg.norm(v)
    m = {"anchor": v.astype(np.float32), "anchor_frac": frac, "size": list(size),
         "orientation": turn, "cluster": 0}
    m["file"] = type("F", (), {"name": f"ref{rng.integers(1 << 30)}.jpg"})()
    return m


def test_consensus_beats_duplicate_outliers(f: Failures) -> None:
    """Three copies of the WRONG candidate, which is what defeats a maximum.

    Taking the single largest similarity asks only whether SOME other reference
    agrees. Duplicates of an impostor agree with each other perfectly, so the
    impostor scored 1.000 and could be drawn as the match panel.
    """
    rng = np.random.default_rng(7)
    right = np.array([1.0] + [0.0] * 15)
    wrong = np.array([0.0, 1.0] + [0.0] * 14)

    real = [candidate(rng, right) for _ in range(5)]
    impostor = candidate(rng, wrong, jitter=0.0)
    dupes = [impostor] + [dict(impostor, file=impostor["file"]) for _ in range(2)]
    items = real + dupes

    info = consensus_match(items)
    score_match(items)

    f.check(info["duplicates"] >= 2,
            f"near-duplicates were not collapsed ({info['duplicates']} found); "
            f"repeated copies can still vote for each other")
    for m in real:
        f.check(m["match"]["verified"],
                "a genuine reference was excluded from the consensus group")
    for m in dupes:
        f.check(not m["match"]["verified"],
                "an impostor backed only by copies of itself was verified; this "
                "is exactly what taking the maximum similarity got wrong")
        f.check(m["match"]["score"] < min(r["match"]["score"] for r in real),
                "an impostor outscored every genuine reference")


def test_consensus_survives_background_and_appearance_changes(f: Failures) -> None:
    """Same candidate, captured in different settings with different attributes.

    Match must not weaken because the scene changed - that was the flaw in
    reading full-image similarity as match or as viewpoint.
    """
    rng = np.random.default_rng(11)
    base = np.array([0.0, 0.0, 1.0] + [0.0] * 13)
    items = [candidate(rng, base, jitter=0.20) for _ in range(6)]
    for i, m in enumerate(items):          # wildly different scenes and attributes
        m["clip"] = np.eye(16)[i].astype(np.float32)
        m["clip_match"] = (base + rng.normal(0, 0.05, 16))
        m["clip_match"] /= np.linalg.norm(m["clip_match"])

    consensus_match(items)
    score_match(items)
    f.check(all(m["match"]["verified"] for m in items),
            "changing background and appearance broke match consensus, which "
            "should depend on the anchor alone")


def test_orientation_is_a_geometric_measurement(f: Failures) -> None:
    """Keypoints, not scene similarity."""
    f.check(anchor_orientation(None) is None and anchor_orientation([(0, 0)]) is None,
            "missing keypoints must yield no orientation rather than a made-up one")

    square_on = [(40, 50), (60, 50), (50, 62), (44, 72), (56, 72)]
    turned = [(40, 50), (60, 50), (57, 62), (44, 72), (56, 72)]
    other = [(40, 50), (60, 50), (43, 62), (44, 72), (56, 72)]
    ys, yt, yo = (anchor_orientation(square_on), anchor_orientation(turned),
                  anchor_orientation(other))
    f.check(abs(ys) < 0.05, f"a square-on anchor reported orientation {ys:+.3f}")
    f.check(yt > 0.2 and yo < -0.2,
            f"turned anchors did not separate by sign ({yt:+.3f}, {yo:+.3f})")

    # Scaling and translating the whole anchor must not change it: it is an
    # orientation, not a position or a size.
    big = [(2 * x + 300, 2 * y + 100) for x, y in turned]
    f.check(abs(anchor_orientation(big) - yt) < 1e-6,
            "orientation changed when the anchor was moved and scaled")


# ---------------------------------------------------------------------------
# multi-candidate references: the failure that tracked the wrong candidate
# ---------------------------------------------------------------------------

class FakeImage:
    """A stand-in for a decoded reference. resolve_targets only needs its size
    and its match; the anchor records are supplied directly."""

    def __init__(self, name, w=1000, h=800):
        self.filename, self.width, self.height = name, w, h

    @property
    def size(self):
        return (self.width, self.height)

    def convert(self, _mode):
        return self


class Quiet:
    def info(self, *a):
        pass
    warning = error = info


def run_resolve(table, exclusions=None):
    """Drive match.resolve_targets over a scripted set of anchor instances."""
    import reference_match as match
    import PIL.ImageOps as IO
    from PIL import Image as PILImage
    real = (match.anchor_instances, PILImage.open, IO.exif_transpose)
    match.anchor_instances = (
        lambda models, pil, detect_candidates=True: [dict(r) for r in table[pil.filename]])
    PILImage.open = lambda p: FakeImage(Path(p).stem)
    IO.exif_transpose = lambda im: im
    try:
        return match.resolve_targets([Path(f"{k}.jpg") for k in table], None,
                                        Quiet(), exclusions)
    finally:
        match.anchor_instances, PILImage.open, IO.exif_transpose = real


def inst(emb, anchor_frac=0.02, candidate=None):
    return {"anchor": np.asarray(emb, np.float32), "box": [0, 0, 50, 50],
            "anchor_frac": anchor_frac, "candidate_box": candidate,
            "keypoints": None, "det_score": 0.9}


def test_wrong_candidate_is_not_the_target(f: Failures) -> None:
    """Two references where the WRONG candidate is larger and more prominent.

    This is the case that put a non-target through four generations. The old rule
    took the largest candidate and the largest anchor per image; here that is the
    wrong candidate in both, and neither of those images shows the target's anchor at
    a size any detector would prefer.
    """
    rng = np.random.default_rng(3)
    target = np.array([1.0] + [0.0] * 15)
    other = np.array([0.0, 1.0] + [0.0] * 14)

    def emb(base, j=0.05):
        v = base + rng.normal(0, j, 16)
        return v / np.linalg.norm(v)

    table = {f"t{i}": [inst(emb(target), 0.02, [0, 0, 200, 400])]
             for i in range(10)}
    for i in range(2):
        table[f"m{i}"] = [
            inst(emb(other), 0.20, [300, 50, 800, 790]),    # larger, in front
            inst(emb(target), 0.001, [10, 100, 120, 500]),  # tiny, the target
        ]

    res = run_resolve(table)
    f.check(res["bank"] is not None and len(res["bank"]) >= 10,
            f"the target bank has "
            f"{0 if res['bank'] is None else len(res['bank'])} anchor(s); the ten "
            f"clean references should all be usable")
    f.check(res["images"] >= 10,
            f"the dominant cluster spans {res['images']} image(s). It must be "
            f"the match in the MOST IMAGES, not the one with the biggest "
            f"anchors - the wrong candidate appears in only two")

    # The wrong candidate's anchor must never enter the bank at all.
    if res["bank"] is not None:
        worst = float(np.max(res["bank"] @ (other / np.linalg.norm(other))))
        f.check(worst < 0.5,
                f"a bank embedding matches the WRONG candidate at {worst:.2f}")

    for i in range(2):
        name = f"m{i}.jpg"
        # These MUST be usable. The two anchors in them are plainly different
        # candidates, so there is nothing ambiguous to refuse - and recovering the
        # target from an image where a non-target is larger is the whole point.
        # Detecting only the largest anchor per image drops the target here and
        # loses the reference entirely.
        if not f.check(name in res["per_image"],
                       f"{name}: the target was not recovered from an image "
                       f"where another candidate is larger; only one anchor per "
                       f"image was considered"):
            continue
        rec = res["per_image"][name]
        chosen = rec["instance"]
        f.check(float(chosen["anchor"] @ target) > float(chosen["anchor"] @ other),
                f"{name}: the LARGER wrong candidate was selected as the target - "
                f"the exact failure that produced a wrong-candidate track")
        f.check(chosen["anchor_frac"] < 0.01,
                f"{name}: selection followed anchor SIZE rather than match")
        f.check(rec["other_candidates"] >= 1,
                f"{name}: the second candidate was not even noticed")
        f.check(rec["candidate_box"] == [10, 100, 120, 500],
                f"{name}: the target anchor was tied to the wrong candidate box, so "
                f"a tracker seeded from it would follow the wrong subject")


def test_ambiguous_image_is_rejected(f: Failures) -> None:
    """Two anchors the embeddings cannot separate - refuse, do not guess."""
    base = np.array([1.0] + [0.0] * 15)
    near = base + np.array([0.0, 0.25] + [0.0] * 14)
    near = near / np.linalg.norm(near)

    res = run_resolve({"a": [inst(base)], "b": [inst(base)],
                       "c": [inst(base), inst(near)]})
    rejected = {n for n, _ in res["rejected"]}
    f.check("c.jpg" in rejected,
            "an image with two indistinguishable anchors was used anyway; with a "
            "second candidate present, guessing is how the wrong one gets tracked")
    f.check({"a.jpg", "b.jpg"} <= set(res["per_image"]),
            "unambiguous images were discarded along with the ambiguous one")


def test_excluded_images_never_reach_the_bank(f: Failures) -> None:
    """The run-specific exclusion list must actually keep them out."""
    target = np.array([1.0] + [0.0] * 15)
    other = np.array([0.0, 1.0] + [0.0] * 14)
    table = {f"t{i}": [inst(target)] for i in range(4)}
    table["w0"] = [inst(other)]
    table["w1"] = [inst(other)]

    res = run_resolve(table, exclusions={"w0.jpg", "w1.jpg"})
    f.check(not ({"w0.jpg", "w1.jpg"} & set(res["per_image"])),
            "an excluded image still contributed to the match bank")
    f.check(len(res["per_image"]) == 4,
            f"{len(res['per_image'])} image(s) usable, expected the 4 kept ones")


def test_exclusions_are_untracked_and_effective(f: Failures) -> None:
    import reference_match as match
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / match.EXCLUSIONS
        p.write_text("# a comment\n\nfoo.jpg\nbar.png  # trailing\n")
        real = match.P.intermediate
        try:
            match.P.intermediate = Path(td)
            got = match.load_exclusions()
        finally:
            match.P.intermediate = real
    f.check(got == {"foo.jpg", "bar.png"},
            f"exclusion parsing produced {got}")


def test_peripheral_extents_are_not_match(f: Failures) -> None:
    """Peripheral extents must not be conditioned from external references.

    How much of one is visible is a fact about the attribute the candidate
    carries. A reference showing more of one instructs the model to expose more,
    which uncovers what the source covers.
    """
    from make_reference_pack import group
    ANCHOR_REGION, COVERING = group("anchor_region"), group("covering")
    ATTRIBUTE, MATCH_ONLY = group("attribute"), group("match_only")
    EXPOSED = group("exposed")

    # Stated as set relations over the GROUPS rather than a list of names. The
    # relations are the stronger claim - they hold for any label map, whereas a
    # name list stops covering whatever the map renames - and they keep category
    # words out of a tracked file.
    f.check(not (MATCH_ONLY & EXPOSED),
            f"exposed class(es) {sorted(MATCH_ONLY & EXPOSED)} are conditioned "
            "from external references; how much of a peripheral extent is "
            "visible is attribute coverage, which belongs to the source")
    f.check(not (MATCH_ONLY & ATTRIBUTE), "an attribute class is conditionable")
    f.check(not (MATCH_ONLY & COVERING),
            "a covering class is conditionable, so an external reference showing "
            "the anchor uncovered could remove the source's covering")
    f.check(MATCH_ONLY <= ANCHOR_REGION,
            f"MATCH_ONLY is {sorted(MATCH_ONLY)}, which reaches outside the "
            "anchor region")
    # Where the map actually has a covering class inside the anchor region, the
    # containment must be STRICT - that is the covered part being withheld.
    # Where it does not, equality is correct and demanding strictness would be
    # asserting a property of one particular label map.
    if COVERING & ANCHOR_REGION:
        f.check(MATCH_ONLY < ANCHOR_REGION,
                "the anchor region has a covering class, so only its uncovered "
                "part may come from an external reference")


def main() -> int:
    # The synthetic binding is injected around the WHOLE suite, before any test
    # that resolves a role. Two things follow, and both are the point:
    #
    #   * these tests run on a fresh checkout, which has no binding file - they
    #     used to require private configuration to execute at all;
    #   * they assert properties of the CODE rather than of one label map, so a
    #     pass here means the authority split holds for any binding, not that
    #     this machine's happens to satisfy it.
    #
    # synthetic_bindings clears every binding-dependent lru_cache on entry and
    # exit, so nothing leaks in from an earlier import or out to a later test.
    with synthetic_bindings():
        return _run(Failures())


def _run(f: Failures) -> int:
    for t in (test_class_partition, test_masking_removes_attributes,
              test_source_panel_is_untouched, test_content_change_invalidates,
              test_composite_key_is_independent_of_generation,
              test_midrun_change_cannot_be_recorded_current,
              test_consensus_beats_duplicate_outliers,
              test_consensus_survives_background_and_appearance_changes,
              test_orientation_is_a_geometric_measurement,
              test_wrong_candidate_is_not_the_target,
              test_ambiguous_image_is_rejected,
              test_exclusions_are_untracked_and_effective,
              test_excluded_images_never_reach_the_bank,
              test_peripheral_extents_are_not_match):
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
