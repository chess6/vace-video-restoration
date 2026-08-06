# inputs/references/

Put your higher-quality stills of the main subject here. **PNG, JPEG or WebP.**

These files are opened read-only. Nothing in this pipeline modifies, renames or
re-encodes them.

## What actually helps

In priority order:

1. **A full-extent shot.** The most valuable single reference. Anchor detail *and*
   extent detail both matter here, and the full extent is what carries attributes,
   silhouette, proportions and accessories.
2. **A clear anchor shot.** As sharp and front-on as you have.
3. **An alternate angle** — from the side or from behind — so the model has
   something to work from when the subject turns away from camera.

## Constraints

- Minimum **256 px** on the short side; smaller images are rejected.
- Images containing **other candidates** are handled: anchors are clustered and
  images whose anchor does not match the dominant one are dropped. Still, prefer
  images where your subject is clearly the subject.
- Near-duplicates are detected automatically and the sharper copy is kept.
- Only **three** views reach the final sheet, because `WanVaceToVideo` consumes
  exactly one reference image (`reference_image[:1]`) and the views are tiled
  into it. Extra references are still useful: they widen the selection pool and
  strengthen the match evidence used for subject tracking.

## Then run

```bash
venv/bin/python scripts/prepare_references.py
```

and inspect the two files it writes (it does not display them):

- `intermediate/reference_sheets/reference_sheet.png` — what VACE will actually see
- `intermediate/reference_sheets/contact_sheet.png` — everything that was considered
