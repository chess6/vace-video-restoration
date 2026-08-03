# inputs/references/

Put your higher-quality stills of the main figure here. **PNG, JPEG or WebP.**

These files are opened read-only. Nothing in this pipeline modifies, renames or
re-encodes them.

## What actually helps

In priority order:

1. **A full-body shot.** The most valuable single reference. Facial *and* body
   detail both matter here, and the full body is what carries clothing,
   silhouette, proportions and accessories.
2. **A clear face shot.** As sharp and front-on as you have.
3. **An alternate angle** — side or back — so the model has something to work
   from when the figure turns away from camera.

## Constraints

- Minimum **256 px** on the short side; smaller images are rejected.
- Photos with **other people** in them are handled: faces are clustered and
  images whose face does not match the dominant identity are dropped. Still,
  prefer photos where your subject is clearly the subject.
- Near-duplicates are detected automatically and the sharper copy is kept.
- Only **three** views reach the final sheet, because `WanVaceToVideo` consumes
  exactly one reference image (`reference_image[:1]`) and the views are tiled
  into it. Extra photos are still useful: they widen the selection pool and
  strengthen the identity match used for subject tracking.

## Then run

```bash
venv/bin/python scripts/prepare_references.py
```

and inspect the two files it writes (it does not display them):

- `intermediate/reference_sheets/reference_sheet.png` — what VACE will actually see
- `intermediate/reference_sheets/contact_sheet.png` — everything that was considered
