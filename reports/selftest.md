# Self-test report

Generated: 2026-08-03T16:05:12-07:00

End-to-end validation of the pipeline on **synthetic** 240p media
(20 s, 4:3, 24 fps, one hard scene cut at 10 s, with an audio tone),
plus three synthetic reference stills.

| Check | Result |
|---|---|
| ComfyUI API responds | PASS |
| generate synthetic 240p source + references | PASS |
| synthetic source clip | PASS |
| inspect_source.py | PASS |
| reports/source_info.json | PASS |
| preprocess_source.py (CFR, scene cuts, 4n+1 chunking) | PASS |
| chunk manifest | PASS |
| manifest invariants (4n+1 lengths, dims %16, cuts respected) | PASS |
| prepare_references.py | PASS |
| reference sheet | PASS |
| contact sheet | PASS |
| make_depth.py | PASS |
| full depth video | PASS |
| track_subject.py (SAM2, manual seed on shot0000) | PASS |
| mask video shot0000 | PASS |
| mask review sheet shot0000 | PASS |
| track_subject.py (SAM2, manual seed on shot0001) | PASS |
| mask video shot0001 | PASS |
| mask review sheet shot0001 | PASS |
| mask alignment (frames/dims/fps match source and depth) | PASS |
| verify_mask_polarity.py (white = regenerate) | PASS |
| extract_pilot.py | PASS |
| run_chunks.py --pilot (real VACE generation) | PASS |
| generated chunk is valid and complete | PASS |
| assemble.py --pilot (audio remux + A/V sync check) | PASS |
| pilot master | PASS |
| master has audio and matching duration | PASS |
| make_comparisons.py | PASS |
| all scripts invocable | PASS |
| run_full.sh refuses without confirmation | PASS |

**Passed: 30 — Failed: 0**

## What this does and does not prove

Proven by this run:

- ffprobe inspection, CFR normalization and square-pixel handling
- scene-cut detection, and that no chunk crosses a cut
- chunk lengths are all 4n+1 and dimensions are multiples of 16
- Depth Anything V2 runs on CUDA and produces frame-aligned depth
- SAM 2 propagates a seed through a shot and exports aligned masks
- mask polarity: white regenerates, black is preserved (measured)
- VACE 1.3B generates a real chunk at the configured size
- chunks assemble, audio is remuxed from the original, A/V sync verified
- run_full.sh refuses to start without explicit confirmation

**Not** proven by this run, and only testable with your real footage:

- match matching quality (Grounding DINO + ArcFace + CLIP ReID).
  A synthetic humanoid is not a candidate; this run seeds SAM 2 manually.
- restoration quality: whether VACE preserves your subject's anchor,
  attributes and proportions convincingly. That is what the real pilot
  and reports/pilot_results.md are for.

Full log: `logs/selftest.log`
