# AGENTS.md — openpilot-nnslr-tools

Rules for any coding agent working in this repository. Read this before making
changes. The project is **advisory-only**: nothing in here is, or may become,
part of a vehicle control path.

## What this repo is (and is not)

- **TRAIN_REPO.** It owns the pure domain contract (`speed_vision_core`),
  dataset tooling (`nnslr_tools`), tests, and docs.
- **Not** the runtime. The Sunnypilot / openpilot integration lives in a
  **separate repository** and is **out of scope** here. Do not reach into or
  modify the runtime from this repo.
- **Two-repo split:**
  - `openpilot-nnslr-tools` (this repo) — dataset prep / training / eval /
    export + the pure `speed_vision_core` contract.
  - `nn-speed-limit-vision` (runtime, a Sunnypilot fork) — **modification
    forbidden** in this phase.

## Hard boundaries (never do these without explicit authorization)

- No **SSH to the comma** device, no remote access of any kind.
- No **downloading** vehicle data or logs.
- No **V100 / GPU** access, no CUDA/PyTorch GPU install, no GPU probe unless the
  task explicitly authorizes one.
- No modification of the **Sunnypilot runtime** / `nn-speed-limit-vision`.
- No **destructive git**: no `reset --hard`, `clean -fd`, `rebase`,
  `filter-repo`, force-push, or history rewrite.
- No **push** unless explicitly asked. Local commits are fine.
- No `speedVisionState` in cereal, no new `carStateSP` consumer, no changes to
  `SpeedLimitResolver` or any control path.

## Authorization model

- Inspecting, analyzing, explaining, comparing, checking syntax, reviewing logs
  or answering a question is **read-only** and needs no further approval.
- Any **modification** (code, config, docs, branches, device state) requires
  explicit authorization for that exact change. Do not add "helpful" refactors,
  cleanup, or safety gates that were not requested.
- When intent is unclear, stop after read-only analysis and ask.

## Module ownership

| Path                       | Owns                                                        | May import                          |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------- |
| `src/speed_vision_core/`   | Pure contract: types, invariants, validation, (de)serial.   | stdlib only. Nothing outside core.  |
| `src/nnslr_tools/`         | CLI + dataset/annotation/split tooling + synthetic fixtures. | `speed_vision_core`, stdlib.        |
| `tests/`                   | CPU, deterministic tests.                                   | both `src` packages, stdlib, pytest.|
| `docs/`                    | Plan, decisions, audit.                                     | —                                   |

`speed_vision_core` must stay importable and runnable with **no third-party
packages**. The import-isolation test
(`tests/cli/test_import_isolation.py`) enforces that banned modules
(`torch`, `cuda`, `nvidia`, `cupy`, `pycuda`, `triton`, `openpilot`, `cereal`,
`selfdrive`, `sunnypilot`, `tinygrad`, `pandas`, `numpy`) are never imported.

## Conventions

- **Topic-tagged block markers** around every change you make, so the diff
  stays visible against upstream across rebases/merges:

  ```python
  # [nnslr-t1] - START
  ...changed code...
  # [nnslr-t1] - END
  ```

  Use one or two lowercase words as the tag. Use the language's comment
  syntax.
- **Small modules, pure functions, explicit schemas, deterministic behavior,
  typed interfaces, reproducible CLI, tests before claiming functionality, no
  hidden global state, no silent coercion of external data, no machine-specific
  paths, no placeholder architecture.**
- **No private data in Git.** Paths, tokens, and machine-specific values stay
  out of the tree. The data root comes from `NNSLR_DATA_ROOT` (decision D4);
  docs use `/path/to/`, never an absolute local path.
- **Stdlib-only** until a task explicitly adds a dependency. Adding one is a
  logged decision and must be reflected in `requirements/` and `pyproject.toml`.
- Keep synthetic fixtures **RFC 8259-valid JSON** (no `NaN`/`Infinity` tokens);
  non-finite *values* are tested in-process, not in documents.

## Test / verify commands

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
nnslr --help
nnslr selftest          # expect: "selftest OK: 18 checks"
pytest -q              # expect: "90 passed"
```

A feature is not "done" until its tests exist and pass on a clean environment.

## Provenance

- Plan / source of truth: `docs/plan.md` (v4).
- Decisions: `docs/vision-speed-limit/decisions.md` (D1–D8).
- T0 audit: `docs/vision-speed-limit/audit.md`.
- Upstream PSA/openpilot operating notes live in the `openpilot_scripts`
  workspace `AGENTS.md`, not here.
