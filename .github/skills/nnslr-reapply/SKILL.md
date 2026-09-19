---
name: nnslr-reapply
description: Rebuild, port, or verify the NNSLR Speed Limit Recognition runtime feature on a current Sunnypilot master while preserving advisory-only isolation.
---

# NNSLR Sunnypilot reapply

Use this skill when asked to update, rebase, port, reconstruct, compare, or
verify NNSLR Speed Limit Recognition against a newer Sunnypilot revision.

## Sources of truth

Read, in this order:

1. `integration/sunnypilot/feature.lock.json`
2. `integration/sunnypilot/docs/runtime-reapply-plan.md`
3. `integration/sunnypilot/hook-map.json`
4. `integration/sunnypilot/protected-surfaces.json`
5. the run receipt produced by `scripts/nnslr_reapply.py`

Do not reconstruct runtime code from chat history.

## Standard workflow

Use `scripts/nnslr_reapply.py`:

```bash
python scripts/nnslr_reapply.py inspect

python scripts/nnslr_reapply.py prepare \
  --upstream-ref refs/heads/master \
  --workspace /path/to/new-run

python scripts/nnslr_reapply.py apply --run-dir /path/to/new-run
python scripts/nnslr_reapply.py verify --run-dir /path/to/new-run --profile source
python scripts/nnslr_reapply.py report --run-dir /path/to/new-run
python scripts/nnslr_reapply.py seal --run-dir /path/to/new-run
```

For offline work, provide local Git sources with `--upstream-repo` and
`--feature-repo`.

Freeze upstream once at prepare time. Never silently switch an in-progress run
to a newer master.

## Mechanical adaptation policy

For a moved import, renamed module, changed hook location, or equivalent API
that preserves the feature contract:

- inspect the new upstream source;
- adapt only the affected integration adapter;
- add/update the regression test;
- continue without asking for permission for every mechanical edit;
- record the adaptation in the report.

Do not use blanket ours/theirs conflict resolution.

A clean cherry-pick is not compatibility proof. Verification and the generated
report are required before source-ready status.

## Blocking compatibility changes

Stop promotion and report the conflict when any of these occur:

- a pinned Cap'n Proto binding is occupied or semantically incompatible;
- accelerator ownership changes so isolation cannot be established;
- the new upstream requires NNSLR to alter vehicle control behavior;
- required timestamp/buffer lifetime guarantees cannot be preserved;
- a protected control surface must be changed;
- required source/build tests cannot be established.

Do not renumber schema automatically.

## Safety and isolation invariants

NNSLR remains advisory-only for this project.

Do not modify or feed Vision into `SpeedLimitResolver` or
`SpeedLimitAssist`.

Do not write `CarStateSP.speedLimit`.

Do not write or impersonate `LiveMapDataSP`.

Do not emit CAN, steering, torque, cruise-button, safety, radar, acceleration,
or actuator requests.

Do not select a different driving model or runner just to make NNSLR work.

Do not initialize Chestnut from a second process.

Default NNSLR mode remains off.

## Device, GPU, and data boundary

Do not access the comma device unless the user explicitly authorizes live
device access for that task.

Do not access GPU hardware or run hardware benchmarks unless explicitly
authorized.

Local comma-origin files are allowed: video, qlog/rlog, replay artifacts and
other files already copied into an authorized local workspace may be processed.
That is not access to the live device.

Do not deploy to the vehicle as part of reapply.

Do not push, force-push, move a stable branch, or publish a release unless that
operation was explicitly requested. In particular, do not force-push a
reconstructed history over a known-good branch.

## Git behavior

Work in the isolated candidate checkout prepared by the tooling.

Preserve the old known-good candidate and its exact commit.

Respect the new upstream gitlinks; never restore old submodule revisions merely
because an old patch expected them.

Do not rewrite the user's normal checkout.

Do not force-push.

If a source patch is empty because upstream appears to contain equivalent work,
do not skip it blindly. Record it as requiring semantic equivalence review.

## Schema behavior

Preserve type IDs, ordinals, units, presence flags, and field meaning.

Do not renumber schema to resolve a conflict.

Old logs must never be decoded using a new meaning for a reused ordinal.

## Result

Every run must distinguish:

- applied source;
- source verification;
- target/hardware verification not run;
- shadow evidence not run;
- advisory release not established.

The report must include the frozen upstream SHA, patch mapping, structural
adaptations, conflicts, test results, not-run tests, result tree, and readiness
level.

A successful source reapply does not authorize deployment.
