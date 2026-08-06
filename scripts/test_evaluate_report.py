#!/usr/bin/env python
"""End-to-end check that evaluate_pilot actually writes its report.

Synthetic video only - a few solid-colour frames from ffmpeg's own generators.
No models, no CUDA, no user media. Attribute parsing is skipped, so nothing here
needs weights.

    venv/bin/python scripts/test_evaluate_report.py

Why this exists: every metric in that script was computed correctly and then
thrown away, because the final statement referenced an undefined name. The
NameError landed after all the work, so a run looked busy, took minutes, and
left no report. Nothing unit-testable was wrong - only the wiring - so the test
has to run the whole script and look for the file.

It runs in a throwaway VACE_RUN namespace, so it cannot touch a real run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = "selftest_evaluate"


def synth(path: Path, colour: str, n: int, w: int = 64, h: int = 48) -> None:
    """A short solid-colour clip. Deterministic and tiny."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={colour}:s={w}x{h}:r=16", "-frames:v", str(n),
         "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)],
        check=True)


def main() -> int:
    if not (ROOT / "venv").exists():
        print("SKIPPED: no venv in this checkout")
        return 0
    env = dict(os.environ, VACE_RUN=RUN)
    run_root = ROOT / "runs" / RUN
    fail: list[str] = []

    with tempfile.TemporaryDirectory():
        inter = run_root / "intermediate"
        reports = run_root / "reports"
        for d in (inter, reports):
            d.mkdir(parents=True, exist_ok=True)

        n = 6
        synth(inter / "source.mkv", "gray", n)
        synth(inter / "mask.mkv", "white", n)      # whole frame is subject
        synth(inter / "variant_a.mkv", "gray", n)
        synth(inter / "variant_b.mkv", "darkgray", n)

        # The minimum evaluate_pilot reads: a manifest with a pilot record, and
        # the variant spec that make_comparisons.py normally writes.
        (inter / "chunk_manifest.json").write_text(json.dumps({
            "pilot": {"start_frame": 0, "end_frame": n},
            "normalized": {"width": 64, "height": 48, "fps": 16,
                           "total_frames": n, "work_path": "x"},
            "shots": [], "chunks": [],
        }))
        (reports / "pilot_variants.json").write_text(json.dumps({
            "source": f"runs/{RUN}/intermediate/source.mkv",
            "mask": f"runs/{RUN}/intermediate/mask.mkv",
            "interval": {"start_frame": 0, "end_frame": n},
            "chunks": ["shot0000_c000"],
            "variants": {
                "a": {"path": f"runs/{RUN}/intermediate/variant_a.mkv",
                      "describes": "identical to source"},
                "b": {"path": f"runs/{RUN}/intermediate/variant_b.mkv",
                      "describes": "darker"},
            },
        }))

        report = reports / "selftest_metrics.json"
        if report.exists():
            report.unlink()
        r = subprocess.run(
            [str(ROOT / "venv/bin/python"), str(HERE / "evaluate_pilot.py"),
             "--no-attribute", "--report", str(report)],
            env=env, capture_output=True, text=True)

        if r.returncode != 0:
            fail.append(f"evaluate_pilot exited {r.returncode}\n"
                        f"{(r.stderr or r.stdout)[-1500:]}")
        elif not report.exists():
            fail.append("evaluate_pilot reported success but wrote no report - "
                        "which is exactly the failure this test exists for")
        else:
            data = json.loads(report.read_text())
            for key in ("pilot", "chunks", "interval", "variants"):
                if key not in data:
                    fail.append(f"report is missing {key!r}")
            got = set(data.get("variants", {}))
            if got != {"a", "b"}:
                fail.append(f"report covers {sorted(got)}, expected ['a', 'b']")
            for name, m in data.get("variants", {}).items():
                if "temporal_stability" not in m:
                    fail.append(f"variant {name} has no metrics in the report")

    if fail:
        print(f"FAILED: {len(fail)} check(s)")
        for m in fail:
            print(f"  - {m}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
