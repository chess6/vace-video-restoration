# githooks/

**Not installed.** `core.hooksPath` is unset at the user's request — nothing runs
on `git push`, and nothing ever ran on `git add` or `git commit` (there was only
ever a `pre-push` hook).

The guards themselves still exist and still work. They are now **manual**:

```bash
scripts/check_repo_clean.sh      # ~8.5s — rules 2a/2c: what would reach the remote
scripts/check_no_display.sh      # rule 1: no viewer, no player, no inline render
```

To put the pre-push hook back:

```bash
git config core.hooksPath githooks
```

## What you are trading

This repository has a public remote, deliberately. `check_repo_clean.sh` is the
last thing standing between the untracked half of this project and that remote,
and it is not theoretical: its needles are derived from the working tree itself,
and checks 6/6b exist because the category has reached a tracked file before.
A leak that reaches a public remote cannot be recalled by deleting the commit.

Run it before a push that touches tracked files. It is one command and it prints
`PASSED — safe to push`.
