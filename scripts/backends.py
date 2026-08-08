"""Which pretrained backend fills each role, loaded from an untracked config.

WHY THIS INDIRECTION EXISTS
CLAUDE.md rule 2a forbids a tracked file from saying what KIND of thing the
subject is. The dependency floor used to exempt third-party model names on the
grounds that they are functional inputs - rename one and the call breaks - and
that exemption was doing real damage, because a pretrained model is named after
what it was trained to find. Naming the backend names the category, in the
clear, in a file every clone carries. The exemption was protecting the API
contract and publishing the thing the rest of the repo withholds.

So the same treatment prompt text already got: the ROLE is tracked, the BINDING
is not. This module resolves a role to a concrete package, model id, revision
and label map, all read from an untracked config on the rule-2c denylist.

WHAT THIS DOES NOT SOLVE, AND WHY THAT IS NOT A REASON TO SKIP IT
requirements.lock.txt and scripts/bootstrap.sh must name a package to install
it, and a lockfile entry cannot be indirected away without breaking the one
guarantee a lockfile makes. That residue is known, accepted and documented. It
is a weaker signal than a model id placed beside a docstring explaining what the
model is FOR - a package name says a capability is installed, not what it is
pointed at - and removing the explanatory half is worth doing even though the
package list stays.

FAIL LOUD, NEVER FALL BACK. A role with no binding raises. The alternative -
quietly substituting some default backend - reproduces the failure that already
cost this project a session: eight LoRA checkpoints scoring identical to four
decimals because a merge had silently no-oped. A pipeline that runs with the
wrong backend produces numbers that look exactly like numbers.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict

_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "backends.local.yaml"
_cache: Dict[str, Any] | None = None


class BackendUnavailable(RuntimeError):
    """A role has no binding, so the stage that needs it cannot run."""


def _load() -> Dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    if not _CONFIG.exists():
        raise BackendUnavailable(
            f"{_CONFIG.relative_to(_CONFIG.parent.parent)} is missing, so no backend "
            "is bound to any role.\n"
            "It is untracked by design (CLAUDE.md rules 2a and 2c): a pretrained "
            "model's name says what it was trained to find, which says what the "
            "subject is.\n"
            "Restore it from a state bundle: scripts/state_bundle.sh import ...\n"
            "There is deliberately no default. A stage that silently ran against "
            "some other backend would emit plausible numbers for the wrong thing."
        )
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise BackendUnavailable(f"PyYAML is required to read {_CONFIG.name}: {exc}")
    with open(_CONFIG) as fh:
        _cache = yaml.safe_load(fh) or {}
    return _cache


def binding(role: str) -> Dict[str, Any]:
    """The config block bound to `role`, or raise.

    Roles used by this pipeline:
      anchor_embed      - resolves the target among candidates (reference_match,
                          track_subject, prepare_references, score_lora_match)
      attribute_parser  - segments the subject's attributes into named regions
                          (make_reference_pack, make_occluders)
    """
    cfg = _load()
    got = cfg.get(role)
    if not isinstance(got, dict) or not got:
        raise BackendUnavailable(
            f"No backend bound to role '{role}' in {_CONFIG.name}.\n"
            f"Add a '{role}:' block. Known roles: anchor_embed, attribute_parser."
        )
    return got


def anchor_embedder(providers=None, log=None):
    """Construct the anchor-embedding backend bound to `anchor_embed`.

    Returns an object with `.get(image)` yielding per-instance records, matching
    what reference_match.py already consumes. Import is deferred to call time so
    that a machine without the package still imports this module.
    """
    b = binding("anchor_embed")
    pkg, model, entry = b.get("package"), b.get("model"), b.get("entrypoint")
    # `entrypoint` is required rather than defaulted. A default would be the
    # constructor's real name, and that name is as category-revealing as the
    # model id this indirection exists to withhold - it would sit in tracked
    # source undoing the whole move, and silently keep working so nothing
    # flagged it.
    if not pkg or not model or not entry:
        raise BackendUnavailable(
            "role 'anchor_embed' needs 'package', 'model' and 'entrypoint' keys"
        )
    providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

    mod = __import__(f"{pkg}.app", fromlist=["*"])
    ctor = getattr(mod, entry)
    app = ctor(name=model, providers=providers)
    app.prepare(ctx_id=0, det_size=tuple(b.get("det_size", (640, 640))))
    if log:
        # The binding is NOT logged. A log line naming the model reintroduces the
        # leak into logs/, which is itself on the denylist for exactly this.
        log.info("Anchor embedding backend ready (%s)", providers[0])
    return app


def attribute_labels() -> Dict[int, str]:
    """id -> label for the attribute parser. Keys are what the model emits."""
    b = binding("attribute_parser")
    labels = b.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise BackendUnavailable("role 'attribute_parser' needs a 'labels' map")
    return {int(k): str(v) for k, v in labels.items()}


def label_group(name: str) -> set:
    """A named subset of the label map: attribute, exposed, match_only, region.

    The groupings are semantic rather than functional - they say which regions
    an external reference may condition - so they live with the labels rather
    than in tracked source, where enumerating them describes the subject.
    """
    b = binding("attribute_parser")
    groups = b.get("groups") or {}
    if name not in groups:
        raise BackendUnavailable(
            f"attribute_parser.groups has no '{name}'. "
            "Expected: attribute, exposed, match_only, anchor_region."
        )
    return set(groups[name])


def env_home(role: str) -> None:
    """Apply any cache-directory env var the bound package needs, if configured."""
    b = binding(role)
    var, val = b.get("home_env"), b.get("home_path")
    if var and val:
        os.environ.setdefault(var, os.path.expanduser(val))
