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

Expected: `nnslr selftest` prints `selftest OK: 18 checks` and `pytest -q`
reports `90 passed`.

> `nnslr env` reports the environment machine-readably. It marks the GPU as
> **untested** and never probes it implicitly — GPU support is only established
> by an explicitly authorized probe (plan §12.4 `check_environment`).

### Data root

Commands that touch real data read the root from the environment variable
**`NNSLR_DATA_ROOT`** — it is **never hardcoded** in code (decision D4). Set it
to the location of your speed-vision data, e.g.:

```sh
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
```

T2 can process comma recordings that are already present locally. It never
connects to the comma device. For qlog/rlog parsing, point the tooling at a
matching local openpilot/sunnypilot checkout:

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

`check-environment` (full), `sync-routes` (remote/device data transfer remains intentionally out of scope);
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
