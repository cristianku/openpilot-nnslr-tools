# openpilot-nnslr-tools

**NNSLR** — *NN Vision Speed Limit Recognition* training and tooling repository.

> **Status: advisory-only.** This project builds an *independent* speed-sign
> perception source for the Swiss driving context. It is **never** part of the
> vehicle control path: it carries no target speed, no actuation request, and no
> cruise set-point. The runtime integration (Sunnypilot / openpilot) lives in a
> **separate repository** and is **out of scope here** (see `AGENTS.md`).

This is the **TRAIN_REPO**. It owns:
- the pure, hardware-independent domain contract (`speed_vision_core`);
- dataset preparation, alignment, annotation, splitting (T2/T3, synthetic-only
  for now);
- training / evaluation / export tooling (T4–T8, not implemented yet).

It deliberately depends on **nothing** beyond the Python standard library at
this stage — no `torch`, no CUDA, no `tinygrad`, no `openpilot`, no `cereal`.

---

## Repository layout

```
src/speed_vision_core/     Pure contract: types, invariants, validation, (de)serialization.
                           Stdlib-only. Imported by nothing outside this repo.
src/nnslr_tools/           CLI (`nnslr`), synthetic fixtures, and (later) data /
                           alignment / annotation / dataset tooling.
tests/core/                Contract tests (pure CPU, deterministic).
tests/cli/                 CLI entry-point tests + import-isolation guard.
src/nnslr_tools/fixtures/  Synthetic ObservationBatch documents (RFC 8259 JSON).
docs/                      Plan, decisions, audit.
requirements/              CPU dependency lock (empty while stdlib-only).
```

---

## Installation on Proxmox GPU container

For a Debian/Ubuntu Proxmox container with NVIDIA passthrough already working
(`nvidia-smi` succeeds), see:

- [docs/proxmox-gpu-container.md](docs/proxmox-gpu-container.md)

It covers cloning NNSLR, ffmpeg/ffprobe, the local qlog/rlog parser environment,
the data-root layout, and the V100-compatible PyTorch CUDA 12.6 environment.

---

## Quickstart (CPU, clean clone)

No GPU, no model, no data. The whole T1 surface runs on the standard library.

```sh
# 1) Create a clean environment (Python >= 3.11).
python3 -m venv .venv
source .venv/bin/activate

# 2) Install editable. T1 has no third-party dependencies.
pip install -e .

# 3) Check the command surface.
nnslr --help

# 4) Run the bundled synthetic self-test (validation + serialization round-trip).
nnslr selftest

# 5) Run the test suite.
pytest -q
```

<!-- [nnslr-sync] - START -->
Expected: `nnslr selftest` prints `selftest OK: 18 checks` and `pytest -q`
passes. Sync integration tests use local fixtures and real rsync, with SSH
transport replaced; they are skipped if rsync is not installed. No test
connects to a comma device.
<!-- [nnslr-sync] - END -->

> `nnslr env` reports the environment machine-readably. It marks the GPU as
> **untested** and never probes it implicitly — GPU support is only established
> by an explicitly authorized probe (plan §12.4 `check_environment`).

### Data root

<!-- [nnslr-sync] - START -->
Set **`NNSLR_DATA_ROOT`** to your private data directory (decision D4), e.g.:
<!-- [nnslr-sync] - END -->

```sh
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
```

<!-- [nnslr-sync] - START -->
`sync-routes` also works without this variable: its default is `/srv/nnslr-data`.
An explicit `--data-root` overrides the variable. Other data-root-dependent
commands still require the variable or their explicit `--data-root` option.

The local processing commands operate on recordings already present on disk.
Only an explicit `sync-routes` invocation connects to the comma. For qlog/rlog
parsing, point the tooling at a matching local openpilot/sunnypilot checkout:
<!-- [nnslr-sync] - END -->

```sh
export NNSLR_OPENPILOT_ROOT=/path/to/openpilot
# Optional when that checkout needs a different Python environment:
export NNSLR_OPENPILOT_PYTHON=/path/to/openpilot-python
```

---

## What is implemented (T1)

| Subcommand        | Purpose                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `nnslr version`   | Print package and core versions.                                        |
| `nnslr env`       | Machine-readable environment report (no implicit GPU probe).            |
| `nnslr validate-batch FILE --now-mono-ns N --expected-session-id S` | Validate an `ObservationBatch` JSON against the §7 contract; stable reason codes; exit `0` accepted / `1` rejected / `2` missing file. |
| `nnslr selftest`  | Run the bundled synthetic fixtures end-to-end (CPU quickstart).         |

The §7 contract (`speed_vision_core.types`) enforces, among other things:
- `value_kph: int | None` — **0 is never a speed limit**; absence stays absent;
  `unknown` / `unreadable` / `not_applicable` / `unavailable` remain distinct.
- Strict numeric deserialization — no silent coercion (`"50"` → `50`, `True` →
  `1`, `50.9` → `50` are all rejected with a deterministic reason code).
- Monotonic timestamps are comparable only within the same session/clock domain.
- A detection must reference the **exact** batch frame (identity + timing +
  native fields), not just the frame key.
- Linked supplementary panels obey the same geometry rules as the primary bbox.
- Deterministic, pure validation reporting a finite set of documented reason codes.
- **No type carries a target speed or actuation request.**

---

<!-- [nnslr-sync] - START -->
## Copy a route from the comma

With an existing SSH alias `comma-remote`, only the route is required:

```sh
nnslr sync-routes --route "$ROUTE"
```

Defaults:

| Setting | Default |
| --- | --- |
| SSH host | `NNSLR_COMMA_HOST`, otherwise `comma-remote` |
| Segments | All available segments of the specified route only |
| Camera | `narrow`: `fcamera.hevc` (front road camera) |
| Log | Full `rlog`, preferring `rlog.zst`, then `rlog`, then `rlog.bz2` |
| Data root | `NNSLR_DATA_ROOT`, otherwise `/srv/nnslr-data` |
| Destination | `<data-root>/raw/routes/<route>/<segment>/` |

Override only what you need:

```sh
# Inspect the remote inventory without downloading or writing local files.
nnslr sync-routes --route "$ROUTE" --dry-run

# Use a different SSH destination, segment selection, or both road cameras.
nnslr sync-routes --route "$ROUTE" --host comma@192.0.2.10 --segments 0-3
nnslr sync-routes --route "$ROUTE" --camera both --data-root /path/to/speed-vision-data

# Set preferences once in your shell configuration instead of repeating options.
export NNSLR_COMMA_HOST=comma-remote
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
```

`--camera wide` selects `ecamera.hevc`, the other forward-facing, wider road
camera. Cabin video and the reduced `qcamera.ts` are not copied. The full log
is retained for frame/capture-time alignment and map-transition candidate
discovery; a `qlog` is not silently substituted for it.

Requires `ssh` and `rsync` locally and `rsync` on the comma. Existing SSH keys,
host verification and `~/.ssh/config` are used; no credentials are stored by
NNSLR. SSH runs in batch mode, so authentication must already work without an
interactive password prompt. No device software is installed or modified.

Rerun the same command to resume: interrupted files stay in `.nnslr-partial/`
until rsync completes them. Existing destination files are checked by rsync
content checksum and reused if identical; different files are reported as
conflicts and never overwritten. Nothing is deleted from either dataset.

Progress is written to stderr. The JSON report on stdout lists selected files,
local SHA-256 hashes, copy/resume/reuse status and missing files/segments. Exit
`0` means success, `1` means missing data or a file transfer/conflict failure,
and `2` means invalid input or a preflight/SSH inventory failure. Available
files can still be copied when other requested data is missing; after a file
transfer failure, remaining entries stay `planned` for the next invocation.
Dry-run contacts the host for inventory, but does not verify existing local
content or transfer files.
<!-- [nnslr-sync] - END -->

## T2 local comma video/log pipeline

These commands operate only on files already copied under local storage. They
do **not** SSH to, download from, or modify a comma device.

| Command | Purpose |
| --- | --- |
| `nnslr route-manifest` | Inventory local route segments, hashes, missing/partial segments. |
| `nnslr inspect-manifest` | Summarize a route manifest. |
| `nnslr video-probe` | Probe fcamera/ecamera/qcamera with ffprobe. |
| `nnslr extract-frames` | Decode with ffmpeg; defaults to 1 fps discovery sampling, with `--all-frames` available. |
| `nnslr log-metadata` | Read local qlog/rlog(.zst) camera/map metadata through a matching local openpilot checkout. |
| `nnslr align-route` | Join ffprobe presentation order to openpilot `EncodeIndex.segmentId`; preserve unresolved frames explicitly. |
| `nnslr alignment-report` | Print deterministic alignment diagnostics. |
| `nnslr find-candidates` | Use map speed transitions as search hints only, never ground truth. |
| `nnslr make-clips` | Build local review clips around projected candidate times. |

The production alignment path deliberately distinguishes:

```text
video PTS / presentation order
        !=
camera frameId
        !=
EncodeIndex.segmentIdEncode (encode order)
        !=
camera capture timestamp
```

Current openpilot defines `EncodeIndex.segmentId` as the index into the camera
file in **presentation order**, so that field is the explicit join key used by
the real local adapter. The old synthetic `manifest.align_frames()` helper
remains fixture-only.

Typical local flow:

```sh
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
export NNSLR_OPENPILOT_ROOT=/path/to/openpilot

nnslr route-manifest raw/routes/000001a3--c20ba54385 \
  --output manifests/routes.jsonl

nnslr video-probe \
  "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc"

nnslr align-route \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --log "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/rlog.zst" \
  --stream narrow_road \
  --segment-num 0 \
  --output "$NNSLR_DATA_ROOT/derived/alignment/segment-0.jsonl"

nnslr find-candidates \
  "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/rlog.zst" \
  --alignment "$NNSLR_DATA_ROOT/derived/alignment/segment-0.jsonl" \
  --output "$NNSLR_DATA_ROOT/manifests/candidates.jsonl"

nnslr make-clips \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --candidates "$NNSLR_DATA_ROOT/manifests/candidates.jsonl" \
  --output "$NNSLR_DATA_ROOT/derived/clips"

nnslr extract-frames \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --output "$NNSLR_DATA_ROOT/derived/frames/discovery"
```

---

## What is not implemented yet (documented, not stubbed)

`nnslr` lists these subcommands and refuses them with exit code `3` so a clean
clone documents what does not exist rather than pretending:

<!-- [nnslr-sync] - START -->
`check-environment` (full);
<!-- [nnslr-sync] - END -->
`import-annotations`, `validate-dataset`, `build-splits` (T3);
`train` (T4), `evaluate` (T5), `mine-hard-examples` (T6),
`export-onnx`, `replay`, `package-model`, `verify-bundle`, `export-core` (T7–T8).

See `docs/plan.md` for the full task breakdown and `docs/vision-speed-limit/`
for decisions (D1–D8) and the T0 audit.

---

## Rules of the road

- **Advisory only** — nothing here may be wired into a control path.
- **No auto-download / train / push / deploy** — those require explicit
  authorization and are gated by task (see `AGENTS.md`).
- **No private data in Git** — paths, tokens, and machine-specific values are
  kept out of the tree (decision D4).
- **Stdlib-only until a task says otherwise** — adding a dependency is an
  explicit, logged change.
