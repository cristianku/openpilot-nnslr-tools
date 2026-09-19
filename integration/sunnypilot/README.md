# Sunnypilot integration / reapply tooling

This directory defines the reproducible integration contract for NNSLR Speed
Limit Recognition on Sunnypilot.

The runtime feature is developed in `cristianku/sunnypilot`; this repository
owns the portable core, feature catalog, compatibility metadata and reapply
tooling. Do not reconstruct runtime code from chat history.

## Bootstrap state

`feature.lock.json` starts in `bootstrap`. That means the integration
metadata can be inspected and a candidate workspace can be prepared, but the
runtime feature is not yet a sealed patchset. `apply` refuses bootstrap locks.

## Commands

```bash
python scripts/nnslr_reapply.py inspect

python scripts/nnslr_reapply.py prepare \
  --upstream-ref refs/heads/master \
  --workspace /tmp/nnslr-port

python scripts/nnslr_reapply.py apply --run-dir /tmp/nnslr-port
python scripts/nnslr_reapply.py verify --run-dir /tmp/nnslr-port --profile source
python scripts/nnslr_reapply.py report --run-dir /tmp/nnslr-port
python scripts/nnslr_reapply.py seal --run-dir /tmp/nnslr-port
```

For offline/reproducible tests, pass `--upstream-repo` and
`--feature-repo` pointing to local Git checkouts or mirrors.

`prepare` freezes the upstream SHA once. A later advance of upstream master
does not alter an in-progress run.

`seal` only creates a local source-ready receipt. It does not push, deploy to
a comma device, or claim hardware/shadow/advisory validation.

See `docs/runtime-reapply-plan.md` for the full R00-R15 plan.
