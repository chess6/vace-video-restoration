#!/usr/bin/env python
"""A missing or malformed backend binding must TERMINATE the stage, not degrade it.

Two things are pinned here, and they are different claims:

  1. FAIL LOUD. `BackendUnavailable` propagates out of the stages that need a
     backend. It used to be caught: prepare_references logged a warning and
     carried on with candidate-box heuristics and no match clustering;
     track_subject logged a warning and flagged every shot for manual seeding.
     Both looked conservative and were not — the stage kept running, produced
     output shaped exactly like correct output, and the step that decides WHICH
     candidate is the target had quietly not happened.

  2. HERMETIC. Importing those stages must not need the binding at all. It is
     untracked by design, so a fresh checkout has none; while roles were
     resolved at module import, `import make_reference_pack` raised and the unit
     tests could not be collected without private configuration.

The distinction that matters and is also pinned: a missing BINDING is fatal, an
ordinary per-image DETECTION FAILURE is not. A backend that loaded and found
nothing in one reference is data; a backend that never loaded makes every such
report meaningless, and conflating them buries a config error inside a warning.

The fixture is invented, so it is category-neutral by construction — it names
roles and fake ids and describes nothing. That is also the only kind of binding
that can live in a tracked file (CLAUDE.md rules 2a, 2c).

    python3 scripts/test_backend_fail_loud.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402

FAILED: list[str] = []

# A complete, entirely synthetic binding. Role words and invented ids only.
FIXTURE = {
    "anchor_embed": {
        "package": "nonexistent_test_package",
        "model": "test-model",
        "entrypoint": "TestApp",
        "det_size": [64, 64],
    },
    "attribute_parser": {
        "id": "test-org/test-parser",
        "revision": "0000000000000000000000000000000000000000",
        "labels": {0: "background", 1: "region_a", 2: "region_b", 3: "region_c"},
        "groups": {
            "attribute": ["region_b"],
            "exposed": ["region_c"],
            "match_only": ["region_a"],
            "anchor_region": ["region_a"],
            "covering": ["region_c"],
        },
    },
}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'ok  ' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def raises_unavailable(fn) -> bool:
    try:
        fn()
    except backends.BackendUnavailable:
        return True
    except Exception:
        return False
    return False


def test_imports_are_hermetic() -> None:
    """Importing a stage must not resolve a binding."""
    backends.use_bindings({})              # no roles bound at all
    try:
        import make_reference_pack
        import make_occluders
        check("make_reference_pack imports with no binding", True)
        check("make_occluders imports with no binding", True)
        # ...and still refuses to return a binding it does not have.
        check("label_map() raises when nothing is bound",
              raises_unavailable(make_reference_pack.label_map))
        check("group() raises when nothing is bound",
              raises_unavailable(lambda: make_reference_pack.group("attribute")))
        check("parser_binding() raises when nothing is bound",
              raises_unavailable(make_reference_pack.parser_binding))
        check("candidate_ids() raises when nothing is bound",
              raises_unavailable(make_occluders.candidate_ids))
    except backends.BackendUnavailable as e:
        check("stages import with no binding", False, f"import raised: {e}")
    finally:
        backends.use_bindings(None)


def test_missing_binding_is_fatal() -> None:
    """Every accessor raises rather than substituting a default."""
    for mapping, label in ((None if False else {}, "empty config"),
                           ({"anchor_embed": {}}, "role present but empty"),
                           ({"attribute_parser": {"labels": {}}}, "empty label map"),
                           ({"attribute_parser": {"labels": {0: "background"},
                                                  "groups": {}}}, "no groups")):
        backends.use_bindings(mapping)
        check(f"binding('attribute_parser') raises — {label}",
              raises_unavailable(lambda: backends.binding("attribute_parser"))
              or raises_unavailable(backends.attribute_labels)
              or raises_unavailable(lambda: backends.label_group("attribute")))
    backends.use_bindings(None)


def test_malformed_binding_is_fatal() -> None:
    """A binding missing a required key is as fatal as no binding."""
    for missing in ("package", "model", "entrypoint"):
        b = dict(FIXTURE["anchor_embed"])
        b.pop(missing)
        backends.use_bindings({"anchor_embed": b})
        check(f"anchor_embedder raises when '{missing}' is absent",
              raises_unavailable(backends.anchor_embedder))
    backends.use_bindings(None)


def test_stages_do_not_swallow_it() -> None:
    """The two stages that used to fall back must let it propagate.

    Asserted against the real functions, with the binding removed underneath
    them, because the bug was a `except Exception` that turned a config error
    into a warning and a None.
    """
    import logging
    log = logging.getLogger("test")
    log.addHandler(logging.NullHandler())

    backends.use_bindings({})
    # A missing THIRD-PARTY dependency skips; a missing BINDING fails. Conflating
    # them is the same mistake this test exists to pin — an absent package says
    # nothing about whether the stage degrades silently, and turning it into a
    # failure would train the reader to ignore a red line.
    try:
        import prepare_references
    except ImportError as e:
        if "backends" in str(e):
            check("prepare_references importable", False, str(e))
        else:
            print(f"SKIP  prepare_references: {e} (environment, not logic)")
    else:
        check("prepare_references.get_anchor_app propagates BackendUnavailable",
              raises_unavailable(lambda: prepare_references.get_anchor_app(log)))

    try:
        import track_subject
        # Construct without running __init__'s heavy work: only .anchor() matters.
        m = track_subject.Models.__new__(track_subject.Models)
        m._anchor = None
        m.log = log
        check("track_subject.Models.anchor propagates BackendUnavailable",
              raises_unavailable(m.anchor))
    except (ImportError, AttributeError) as e:
        check("track_subject importable with an anchor() method", False, str(e))
    finally:
        backends.use_bindings(None)


def test_fixture_is_usable() -> None:
    """A synthetic binding satisfies every accessor, so tests can run on a clone."""
    backends.use_bindings(FIXTURE)
    try:
        import make_reference_pack as mrp
        mrp.label_map.cache_clear()
        mrp.group.cache_clear()
        mrp.parser_binding.cache_clear()
        check("label_map() resolves from the fixture",
              mrp.label_map() == {0: "background", 1: "region_a",
                                  2: "region_b", 3: "region_c"})
        check("group() resolves from the fixture",
              mrp.group("match_only") == {"region_a"})
        check("parser_binding() resolves from the fixture",
              mrp.parser_binding()["revision"].startswith("0000"))
        import make_occluders as mo
        mo.candidate_ids.cache_clear()
        check("candidate_ids() excludes background", mo.candidate_ids() == {1, 2, 3})
        check("an unknown group still raises",
              raises_unavailable(lambda: backends.label_group("no_such_group")))
    finally:
        for mod, names in (("make_reference_pack",
                            ("label_map", "group", "parser_binding")),
                           ("make_occluders", ("candidate_ids",))):
            m = sys.modules.get(mod)
            for n in names:
                if m and hasattr(getattr(m, n, None), "cache_clear"):
                    getattr(m, n).cache_clear()
        backends.use_bindings(None)


def main() -> int:
    test_imports_are_hermetic()
    test_missing_binding_is_fatal()
    test_malformed_binding_is_fatal()
    test_stages_do_not_swallow_it()
    test_fixture_is_usable()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all backend fail-loud checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
