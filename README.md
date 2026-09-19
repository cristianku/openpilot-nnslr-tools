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
<!-- [route-extract] - START -->
`sync-routes` and `extract-frames --route` also work without this variable:
their default is `/srv/nnslr-data`. An explicit `--data-root` overrides the
variable. Other data-root-dependent commands still require the variable or
their explicit `--data-root` option.
<!-- [route-extract] - END -->

The local processing commands operate on recordings already present on disk.
Only an explicit `sync-routes` invocation connects to the comma. For qlog/rlog
parsing, point the tooling at a matching local openpilot/sunnypilot checkout:
<!-- [nnslr-sync] - END -->

```sh
export NNSLR_OPENPILOT_ROOT=/path/to/openpilot
# The child interpreter needs only the optional log-reader dependencies:
export NNSLR_OPENPILOT_PYTHON=/path/to/parser-venv/bin/python
# Only if compatible car.capnp is not embedded in the checkout:
export NNSLR_OPENDBC_ROOT=/path/to/opendbc
```

<!-- [schema-reader] - START -->
Create the optional parser environment once, separately from the core CLI:

```sh
python3 -m venv /path/to/parser-venv
/path/to/parser-venv/bin/python -m pip install -r requirements/log-reader.txt
```

The reader loads `openpilot/cereal/log.capnp` (or `cereal/log.capnp`) directly
in a child process; it does not import or install the openpilot runtime.
`log-metadata`, `align-route` and `find-candidates` also accept
`--openpilot-root`, `--openpilot-python` and `--opendbc-root` overrides.
Use compatible `log.capnp` and `car.capnp` versions: a schema mismatch or
truncated log fails the command instead of returning partial metadata.
<!-- [schema-reader] - END -->

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

With an existing SSH alias `comma`, only the route is required:

```sh
nnslr sync-routes --route "$ROUTE"
```

Defaults:

| Setting | Default |
| --- | --- |
| SSH host | `NNSLR_COMMA_HOST`, otherwise `comma` |
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
export NNSLR_COMMA_HOST=comma
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

<!-- [route-extract] - START -->
## Extract images from a complete route

Once a route has been downloaded, one command processes every available local
segment automatically. No shell loop or segment list is needed:

```sh
nnslr extract-frames --route "$ROUTE"
```

Defaults are `fcamera.hevc` (front narrow road camera), one PNG per second at
the original resolution, and `NNSLR_DATA_ROOT` or `/srv/nnslr-data` as the data
root. Images and a `frames.jsonl` manifest are written for each segment:

```text
<data-root>/derived/frames/<route>/<segment>/frame_00000000.png
<data-root>/derived/frames/<route>/<segment>/frames.jsonl
```

The usual download-to-images workflow is therefore:

```sh
nnslr sync-routes --route "$ROUTE"
nnslr extract-frames --route "$ROUTE"
```

Only specify options when changing a default:

```sh
nnslr extract-frames --route "$ROUTE" --fps 5
nnslr extract-frames --route "$ROUTE" --all-frames
nnslr extract-frames --route "$ROUTE" --data-root /path/to/speed-vision-data
nnslr extract-frames --route "$ROUTE" --output /path/to/images
```

`--output` is the parent of the per-segment directories. `--manifest FILE`
optionally adds one combined JSONL in addition to the per-segment manifests.
`--start` and `--end`, when supplied, apply within **each segment**, not to a
route-wide clock. Segment discovery follows the local folders created by
`sync-routes`; missing indices are not renumbered. A missing `fcamera.hevc`
or pre-existing frame output is reported before extraction starts. Existing
images are not overwritten unless `--overwrite` is explicitly supplied.

This step extracts images for review and annotation. It does not recognize
signs. Capture timestamps are attached only when a matching, hash-bound
video/log alignment is available; otherwise timing remains explicitly unresolved. Single-video processing remains
available with `--video FILE --output DIRECTORY`.
<!-- [route-extract] - END -->

<!-- [preannotation] - START -->
## Pretrained proposals and visual review (CPU)

Install the optional inference dependencies once, from this repository in the
active nnslr virtual environment:

```sh
python -m pip install '.[preannotate]'
```

After extracting a route, these commands process all its frames and open the
review server. No environment variables, output directories or segment loop
are required; the default data root is `/srv/nnslr-data`.

```sh
nnslr preannotate --route 00000089--0ac1c0fdec
nnslr review --route 00000089--0ac1c0fdec
```

Open `http://127.0.0.1:8765` in the browser. When running in the container, start
review from your Mac through an SSH tunnel instead:

```sh
ssh -L 8765:127.0.0.1:8765 openpilot-nnslr 'bash -lic "nnslr review --route 00000089--0ac1c0fdec"'
```

Then open the same local browser address. The server binds to loopback by
default and serves only the report and its image previews. Stop with Ctrl+C.

The first preannotation run downloads checksum-pinned ONNX weights under
`models/preannotation/` inside the data root; subsequent runs use that cache.
Inference uses CPU only in an isolated subprocess. Every run preserves the
previous results and writes a new directory under `derived/preannotations/`.
The review command selects the latest completed run automatically.

The page shows native-resolution boxes and proposed values. You can change
the type/value, delete false positives, draw missed boxes, and confirm each
frame. Edits persist in that browser. **Download the reviewed JSONL** to keep
them outside browser storage. Only explicitly confirmed frames are exported;
a frame with no proposals is never automatically a negative training example.
Exports retain the original proposals and remain `training_ready: false` until
`import-annotations` validates them. Missing capture timing remains explicitly
unknown: reviewed images may support classification, but not passage/latency claims.
The review page records a reviewer name/pseudonym and confirmation time; reopening
`nnslr review` updates older report pages without rerunning the model.

### What the models recognize

- Vertical signs: a pretrained 82-class Vietnam sign detector, applied to the
  whole frame plus overlapping native-resolution tiles. Swiss accuracy is
  **not established**. A numeric OCR reading is not used as a vertical sign.
- Asphalt numbers: PP-OCRv4 reads the lower half with 3x vertical expansion.
  A separate Cityscapes SegFormer model identifies the road surface. At least
  95% of the number box **including surrounding context** must lie on pixels
  classified as road with score at least 0.9. Missing/uncertain road evidence
  rejects the reading. Wall, advertising, vehicle and sidewalk regions are
  excluded when segmentation identifies them correctly.
- An accepted asphalt number is a `road_marking_candidate`, not an established
  speed limit. OCR score is not a probability of being a speed sign. Surface
  segmentation can still fail; arbitrary numbers painted on roads, adjacent
  roads and ambiguous markings require human review. This is dataset
  preparation, with no vehicle-control or runtime integration.

### Model provenance and usage constraints

Model repositories, revisions, checksums, preprocessing and provenance are
recorded in each run and each frame. External weights and private images are
not committed to this repository.

| Component | Source | License information |
| --- | --- | --- |
| Sign detector | [star092304/traffic-sign-detection-vietnam-yolo](https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo) | Model card says Apache-2.0; embedded ONNX metadata says AGPL-3.0. This discrepancy is recorded, not resolved by this tool. |
| Road segmentation | [Xenova SegFormer ONNX](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-1024-1024), from NVIDIA SegFormer | [NVIDIA license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) limits use to non-commercial research/evaluation. |
| OCR | [RapidOCR 3.4.0](https://pypi.org/project/rapidocr/3.4.0/) bundled PP-OCRv4 | Apache-2.0; bundled model hashes are checked before use. |

These are research/evaluation aids for producing reviewed annotations, not a
validated model bundle for deployment. The pretrained components have their
own terms; the repository's MIT license does not replace them.
<!-- [preannotation] - END -->

## T2 local comma video/log pipeline

These commands operate only on files already copied under local storage. They
do **not** SSH to, download from, or modify a comma device.

<!-- [route-extract] - START -->
| Command | Purpose |
| --- | --- |
| `nnslr route-manifest` | Inventory local route segments, hashes, missing/partial segments. |
| `nnslr inspect-manifest` | Summarize a route manifest. |
| `nnslr video-probe` | Probe fcamera/ecamera/qcamera with ffprobe. |
| `nnslr extract-frames` | Use `--route ID` for all local segments automatically, or `--video FILE --output DIR` for one video; defaults to 1 fps, with `--all-frames` available. |
| `nnslr log-metadata` | Read local qlog/rlog(.zst) camera/map metadata through a matching local openpilot checkout. |
| `nnslr align-route` | Join ffprobe presentation order to openpilot `EncodeIndex.segmentId`; preserve unresolved frames explicitly. |
| `nnslr alignment-report` | Print deterministic alignment diagnostics. |
| `nnslr find-candidates` | Use map speed transitions as search hints only, never ground truth. |
| `nnslr preannotate` | Propose sign boxes and road-surface-gated asphalt numbers on extracted route frames, using optional CPU models. |
| `nnslr review` | Review the latest route proposals in a local browser and export confirmed annotations. |
| `nnslr import-annotations` | Validate and import confirmed review JSONL; defaults to the annotation inbox and canonical objects file. |
| `nnslr validate-dataset` | Check schema, images, hashes, boxes, review/provenance and timing; optionally verify frozen splits. |
| `nnslr build-splits` | Freeze route/site/encounter/image-connected partitions with deterministic defaults and leakage checks. |
| `nnslr make-clips` | Build local review clips around projected candidate times. |
<!-- [route-extract] - END -->

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

<!-- [t2-validation] - START -->
The adapter uses `EncodeIndex.segmentId` as its presentation-order join key
only when the complete index domain matches decoded positions `0..N-1`.
Real logger startup can leave a nonzero first `segmentId`; truncated or missing
data can also break that domain. Those segments remain unresolved with
`presentation_index_domain_mismatch` (CLI exit 1); the tool never silently
subtracts an offset. Duplicate indices and missing capture timestamps also
remain unresolved. The synthetic `manifest.align_frames()` helper is fixture-only.

Raw HEVC often has no PTS. The frame/alignment output exposes a separate
`media_time_s` / `video_media_time_s` and provenance
`raw_hevc_frame_duration` when every decoded frame has a valid duration.
PTS stays null; camera capture time still comes only from the log. Other
containers, mixed timestamps or missing durations do not use that fallback.

`make-clips` automatically reencodes raw `.hevc`/`.h265` to H.264 MP4 and
performs seeking after opening the input; packet-copy seeks produced empty
files on real recordings. `extract-frames --start` also decodes before seeking.
These operations can take longer because they decode from the beginning.
Empty output is an error. For timestamped containers the default clip mode is
packet copy (keyframe boundaries); use `--reencode` for a precise cut.

See [the real-route validation report](docs/T2_REAL_ROUTE_VALIDATION.md) for
verified coverage and outstanding T2 checks.
<!-- [t2-validation] - END -->

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

<!-- [route-extract] - START -->
For the complete option list of any implemented command, use its built-in help:

```sh
nnslr --help
nnslr sync-routes --help
nnslr extract-frames --help
```
<!-- [route-extract] - END -->

<!-- [reviewed-dataset] - START -->
## Import reviewed annotations and build splits

After confirming frames and downloading the JSONL from the review page:

```sh
nnslr import-annotations annotations-RUN_ID.jsonl
nnslr validate-dataset
nnslr build-splits
```

No environment variables or output folder arguments are required. All three
commands use the configured data root, defaulting to `/srv/nnslr-data`.
The default canonical dataset is `annotations/objects.jsonl`. Alternatively,
put review files into `annotations/inbox/` under that root and run
`nnslr import-annotations` with no arguments. Multiple input files can be
imported together; reimporting duplicate identities is rejected.

`build-splits` creates an immutable content-addressed JSON under `splits/`, with
`latest.json` pointing to it. Seed defaults to `0`. It requires at least five
independent connected groups and a reviewed negative group to populate train,
validation, test, route-held-out and hard-negative. A route is never split by
frame; known shared sites, encounters and duplicate images also stay together.
With only two routes, this command reports insufficient groups instead of
manufacturing an apparently independent test set. Unknown repeated physical
sites remain a reported grouping limitation.

An invalid dataset returns a JSON error report and nonzero exit code. For an
existing split, `nnslr validate-dataset --splits /path/to/split.json` also checks
membership, dataset identity and leakage. See [the canonical dataset contract](docs/DATASET.md).

Extraction now records source decoded index, native geometry, image/video hashes,
media time and provenance. Sampling picks the first available source frame at
each requested interval and never duplicates it. `output_index` remains a
separate PNG counter. This may retain one more frame at the end of a segment
than the previous ffmpeg FPS filter. `--overwrite` stages and validates the new
images before replacing old images and removing stale PNGs.

`extract-frames --alignment FILE` attaches capture provenance for a single
video; with `--route`, pass a directory containing `<segment>.jsonl` files.
Route mode also discovers these files at `derived/alignment/ROUTE_ID/` by
default. Generate alignments with the current `align-route` command: older
files without source hashes are rejected rather than trusted implicitly.
<!-- [reviewed-dataset] - END -->

## What is not implemented yet (documented, not stubbed)

`nnslr` lists these subcommands and refuses them with exit code `3` so a clean
clone documents what does not exist rather than pretending:

<!-- [nnslr-sync] - START -->
`check-environment` (full);
<!-- [nnslr-sync] - END -->
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
