#!/usr/bin/env python
"""A synthetic backend binding, shared by every test that needs one.

WHY IT IS INVENTED RATHER THAN A COPY. The real binding is untracked (CLAUDE.md
rules 2a and 2c) because a pretrained model is named for what it was trained to
find, and its label map enumerates that class's parts. A fixture derived from it
would carry the same information into a tracked file. This one is made up: role
words, fake ids, region_a/b/c. It describes nothing, which is precisely why it
can be committed.

WHY IT IS SHARED. Each test that inlines its own fixture invents a slightly
different label map, and a test then passes or fails on a detail of its own
invention rather than on the code. One fixture, one shape.

THE CACHE PROBLEM, which is the whole reason this is a context manager rather
than a dict. Every binding-dependent accessor is lru_cached, deliberately - a
stage resolves its role once. That makes them stateful across tests: a test that
resolves `group("attribute")` under the fixture leaves the fixture's answer
cached, and the NEXT test sees it whether it injected anything or not. Caches are
therefore cleared on the way IN as well as on the way out. Clearing only on exit
leaves whatever the process resolved before the test ever started.

    from test_fixtures import synthetic_bindings

    with synthetic_bindings():
        ...                      # every role resolves to the fixture

    with synthetic_bindings(None):
        ...                      # every role raises, as on a fresh clone
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402

# Entirely synthetic. Role words and invented ids; no real model, no real label.
FIXTURE: dict = {
    "anchor_embed": {
        "package": "nonexistent_test_package",
        "model": "test-model",
        "entrypoint": "TestApp",
        "det_size": [64, 64],
    },
    "attribute_parser": {
        "id": "test-org/test-parser",
        "revision": "0" * 40,
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

# Every lru_cached accessor that resolves a binding, as (module, attribute).
# Modules are named rather than imported at module scope: importing a stage here
# would defeat the point, since the suite must be collectable without a binding.
_CACHED_ACCESSORS = (
    ("make_reference_pack", "label_map"),
    ("make_reference_pack", "group"),
    ("make_reference_pack", "parser_binding"),
    ("make_occluders", "candidate_ids"),
)


def clear_binding_caches() -> list[str]:
    """Clear every binding-dependent lru_cache that is currently imported.

    Only touches modules already in sys.modules: importing one to clear its cache
    would import a stage the caller may deliberately not have loaded.
    """
    cleared = []
    for mod_name, attr in _CACHED_ACCESSORS:
        mod = sys.modules.get(mod_name)
        fn = getattr(mod, attr, None) if mod else None
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
            cleared.append(f"{mod_name}.{attr}")
    return cleared


@contextlib.contextmanager
def synthetic_bindings(mapping: dict | None = FIXTURE):
    """Bind every role to the fixture for the duration of the block.

    Pass `None` for the fresh-clone case: nothing bound, every role raises.
    Caches are cleared entering AND leaving, so neither the block nor whatever
    follows it inherits a resolution from the other.
    """
    clear_binding_caches()
    backends.use_bindings(mapping if mapping is not None else {})
    try:
        yield mapping
    finally:
        backends.use_bindings(None)
        clear_binding_caches()


if __name__ == "__main__":       # a smoke test of the helper itself
    with synthetic_bindings():
        assert backends.attribute_labels()[1] == "region_a"
        assert backends.label_group("match_only") == {"region_a"}
    with synthetic_bindings(None):
        try:
            backends.attribute_labels()
            raise SystemExit("FAIL: unbound roles must raise")
        except backends.BackendUnavailable:
            pass
    print("test_fixtures: synthetic binding injects and unbinds correctly")
