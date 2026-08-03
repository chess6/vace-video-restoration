# githooks/

`.git/hooks/` is not versioned, so a fresh clone has no guards. Install them:

```bash
git config core.hooksPath githooks
```

`pre-push` then runs `scripts/check_repo_clean.sh` and
`scripts/check_no_display.sh` before every push and blocks the push if either
fails. See `CLAUDE.md` rules 1 and 2a for what they enforce.
