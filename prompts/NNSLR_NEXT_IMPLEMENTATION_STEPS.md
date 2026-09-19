# NNSLR — Next Implementation Steps

**Date:** 2026-09-19  
**Scope:** `openpilot-nnslr-tools` + `cristianku/sunnypilot` runtime integration.

<!-- [reviewed-dataset] - START -->
## Local implementation update (2026-09-19)

- A1: segment 0 is formally unresolved; no index offset is guessed.
- A2: ten visual inspection points are recorded, but independent measured
  timing errors/human synchronization audit are still missing. T2 stays partial.
- A3: extracted manifests now preserve decoded index, geometry, media/capture
  provenance and source hashes; current hash-bound alignments can be attached.
- A4: additional decoder/stream/empty/duplicate-directory regressions exist.
- B1–B4: canonical schema, `import-annotations`, `validate-dataset` and
  deterministic connected-group `build-splits` are implemented locally.
- Browser review records reviewer/time and optional site/encounter labels;
  original proposals and rejected records remain available for audit.
- The reviewed Swiss corpus, trained baseline, evaluation/export bundle and
  runtime integration milestone remain pending. Synthetic tests are not a
  substitute for human annotations or independent synchronization evidence.

See [DATASET.md](../docs/DATASET.md) and the updated
[T2 report](../docs/T2_REAL_ROUTE_VALIDATION.md). The historical snapshot below
is retained for provenance; source and these updates supersede its CLI list.
<!-- [reviewed-dataset] - END -->

## 0. Verified current state

### TRAIN_REPO

Repository:

```text
cristianku/openpilot-nnslr-tools
```

Observed `main`:

```text
cd68edf0c7577850c13d637bd3d07904f945ce82
```

Latest observed CI: **green**.

Implemented CLI:

```text
nnslr version
nnslr selftest
nnslr sync-routes
nnslr route-manifest
nnslr inspect-manifest
nnslr video-probe
nnslr extract-frames
nnslr log-metadata
nnslr align-route
nnslr alignment-report
nnslr find-candidates
nnslr make-clips
nnslr preannotate
nnslr review
```

Still not implemented:

```text
import-annotations
validate-dataset
build-splits
train
evaluate
mine-hard-examples
export-onnx
replay
package-model
verify-bundle
export-core
```

### Sunnypilot runtime

Repository/branch:

```text
cristianku/sunnypilot:nn-speed-limit-vision
```

Observed HEAD:

```text
de14a08d5efe9007d00857513e07e734fad614b3
```

Observed upstream base:

```text
sunnypilot/sunnypilot:master
a5f44653d7f43ad57fef2f546f3916ec4cbf3c56
```

Already present:

```text
openpilot/sunnypilot/speed_vision/
speedVisionState
VisionSpeedLimitMode
VisionSpeedLimitWarnings
VisionSpeedLimitDiagnostics
speedvisiond
separate VISION HUD
```

Current runtime artifact state:

```text
model.state = unavailable
bundle_digest = null
```

So the runtime shell exists, but no real NNSLR model is running yet.

### Reapply tooling

Branch:

```text
cristianku/openpilot-nnslr-tools:nnslr/sunnypilot-reapply-v1
```

Observed HEAD:

```text
e69876d1ec81f2f792c3a45fdd3525986b7243b7
```

Latest observed CI: **green**.

Current lock:

```text
state = bootstrap
source_feature_commit = null
patches = []
```

The reapply mechanism exists but is not sealed.

---

# 1. Permanent architecture rules

NNSLR is **advisory-only**.

Do not:

```text
write CarStateSP.speedLimit
write or impersonate LiveMapDataSP
feed Vision into SpeedLimitResolver
feed Vision into SpeedLimitAssist
send CAN
change cruise target
change acceleration
change steering/torque
change safety configuration
```

Sources stay independent:

```text
CAR
MAP
VISION
```

For the first release, only CAR/MAP remain operational SLA sources.

TRAIN_REPO owns dataset/training/evaluation/export/core.

SUNNYPILOT owns camera input/runtime inference/messaging/UI.

---

# 2. Phase A — close the T2 real-route gate

Current report:

```text
docs/T2_REAL_ROUTE_VALIDATION.md
```

T2 is still partial.

## A1. Resolve or formally document segment-0 alignment

Observed real behavior:

```text
decoded frames: 1180
EncodeIndex.segmentId: 20..1199
```

Current rejection:

```text
presentation_index_domain_mismatch
```

Next action:

- inspect verified producer/packet behavior;
- determine why the initial index domain starts at 20;
- never fix this by blindly subtracting 20.

Acceptance:

```text
verified mapping established
OR
segment remains explicitly unresolved
```

## A2. Complete ten-point manual sync audit

For at least 10 representative points record:

```text
route_id
segment
decoded frame index
video media time
capture timestamp
candidate timestamp
visual event
measured timing error
alignment provenance
pass/fail/unresolved
```

## A3. Enrich extracted-frame manifests

Add when available:

```text
route_id
segment_index
output_index
decoded_frame_index
camera_stream
width
height
media_time_s
media_time_provenance
capture_mono_ns
capture_time_provenance
alignment_status
alignment_reason
source_video_sha256
source_log_sha256
```

Important:

```text
output_index != decoded_frame_index
```

## A4. Add T2 edge-case coverage

Test:

```text
partial GOP
truncated HEVC
zero-frame video
missing video
missing/truncated rlog
ecamera.hevc
qcamera.ts
segment gaps
duplicate segment directories
```

No corrupted or unresolved input may silently become dataset ground truth.

---

# 3. Phase B — reviewed annotation dataset

Preannotation now exists, but it is **not ground truth**.

Pipeline:

```text
frames
  ↓
preannotation
  ↓
HTML review
  ↓
human-confirmed annotations
```

## B1. Freeze canonical annotation schema

Minimum fields:

```text
schema_version
route_id
segment_index
decoded_frame_index
capture_mono_ns
camera_stream
image_path
image_sha256

bbox_xyxy
sign_family
value_state
value_kph

applicability
supplementary_panel
visibility
occlusion
temporary
variable_display

proposal_source
proposal_score
review_state
reviewer
review_timestamp
notes
```

Enums should include:

```text
sign_family:
  maximum_speed
  cancellation
  zone
  variable_display
  other_sign
  not_a_sign

value_state:
  read
  unreadable
  unknown
  not_applicable

applicability:
  own_road
  other_road
  unresolved

review_state:
  pending
  accepted
  corrected
  rejected
```

## B2. Implement `nnslr import-annotations`

Requirements:

- strict schema validation;
- bbox validation;
- duplicate rejection;
- provenance preservation;
- preannotation only becomes dataset truth after explicit review.

## B3. Implement `nnslr validate-dataset`

Detect:

```text
missing image
hash mismatch
invalid bbox
illegal speed
unknown enum
duplicate identity
unreviewed proposal
capture-time inconsistency
missing provenance
split leakage risk
```

Return a machine-readable report and non-zero exit on invalid data.

## B4. Implement `nnslr build-splits`

Required splits:

```text
train
validation
test
route-held-out
hard-negative
```

Do not random-split neighboring frames.

Split by route/sequence/event boundaries to avoid leakage.

---

# 4. Phase C — first NNSLR baseline model

Goal: produce a measurable baseline, not the final production model.

## C1. Freeze the model I/O contract first

Define:

```text
input size
dtype
color order
normalization
crop/letterbox
detector output schema
reader output schema
class vocabulary
confidence fields
```

Recommended architecture:

```text
road frame
  ↓
detector
  ↓
sign crop
  ↓
reader/classifier
  ↓
ObservationBatch evidence
```

The network must not directly output an operational target speed.

## C2. Implement `nnslr train`

Record:

```text
dataset version/hash
git commit
seed
architecture
optimizer
learning rate
batch size
epochs
checkpoint hashes
metrics
```

V100:

```text
FP16 supported
BF16 not supported
```

## C3. Train first baseline on reviewed Swiss data

Collect:

```text
training duration
GPU utilization
VRAM
loss curves
validation metrics
checkpoint hash
```

---

# 5. Phase D — evaluation and hard-example mining

Implement:

```text
nnslr evaluate
nnslr mine-hard-examples
```

Detector metrics:

```text
precision
recall
mAP
false positives/hour
false negatives/hour
```

Reader metrics:

```text
numeric accuracy
sign-family accuracy
unreadable accuracy
unknown rejection
```

End-to-end:

```text
correct sign + value
time-to-first-detection
stable-value latency
other-road false positive rate
unsupported-domain rejection
```

Important confusion pairs:

```text
30/80
50/60
60/80
80/100
100/120
```

Hard examples must return to the reviewed dataset. Do not patch model errors with ad-hoc runtime rules.

---

# 6. Phase E — export and verified model bundle

Implement:

```text
nnslr export-onnx
nnslr package-model
nnslr verify-bundle
nnslr export-core
```

Expected logical bundle:

```text
speed_limit_recognition/
├── detector.*
├── reader.*
├── config.json
├── classes.json
├── capability.json
├── model_card.json
└── sha256sums.json
```

Freeze:

```text
input dimensions
dtype
color format
normalization
crop/letterbox
output tensor names
class vocabulary
schema version
model version
dataset version
training commit
hashes
```

Before Sunnypilot integration, establish export parity:

```text
training framework output
≈
exported model output
```

on fixed golden fixtures.

---

# 7. Phase F — fix Sunnypilot CI before adding inference

Current observed runtime CI:

```text
22 passed
6 failed
```

The six observed failures are repo-root path errors in tests, not demonstrated runtime logic failures.

Affected files include:

```text
test_control_isolation.py
test_faults.py
test_schema_roundtrip.py
test_ui_isolation.py
```

Expected repository root in Actions:

```text
/home/runner/work/sunnypilot/sunnypilot
```

Fix the test root calculation and require:

```bash
PYTHONPATH=. pytest -q openpilot/sunnypilot/speed_vision/tests
```

to be fully green before wiring a real model.

---

# 8. Phase G — integrate the verified model into Sunnypilot

## G1. Update `artifacts.lock.json`

Only after a verified bundle exists.

Current:

```text
model.state = unavailable
bundle_digest = null
```

Target logical state:

```text
model.state = verified
bundle_digest = <real sha256>
bundle_version = <version>
```

Never commit a fake digest.

## G2. Add artifact loader

Recommended module:

```text
openpilot/sunnypilot/speed_vision/artifacts.py
```

Responsibilities:

```text
locate bundle
verify sha256
load config
load classes
check schema version
check expected files
fail closed
```

## G3. Add runtime preprocessing

Recommended:

```text
preprocessing.py
```

Pipeline:

```text
NV12
→ RGB/BGR
→ crop/letterbox
→ resize
→ normalize
→ tensor layout
```

Rules must come from the model bundle.

Training preprocessing and runtime preprocessing must match on golden fixtures.

## G4. Connect `speedvisiond` to VisionIPC

Path:

```text
camerad
  ↓
VisionIpcClient
  ↓
VISION_STREAM_NARROW_ROAD
  ↓
adapter
  ↓
preprocessing
  ↓
runner
```

Required behavior:

```text
connect/reconnect
camera restart handling
capture timestamp preservation
frame identity preservation
pending queue <= 1
drop stale pending frame
explicit unavailable state
```

## G5. Select inference ownership

Option A:

```text
speedvisiond → independent inference backend
```

Option B, if accelerator ownership requires it:

```text
modeld
  ├── driving model
  └── NNSLR inference
          ↓
speedVisionObservations
          ↓
speedvisiond
```

If Chestnut is used:

```text
DO NOT create a second Chestnut owner
```

## G6. Implement the real runner

Required interface:

```python
prepare()
submit_latest(frame)
poll()
health()
close()
```

`prepare()`:

```text
verify model bundle
initialize backend
warm up off-road
fail closed
```

`submit_latest()`:

```text
newest frame wins
no unbounded queue
```

`poll()`:

```text
returns ObservationBatch
preserves original capture identity/time
```

`health()`:

```text
available
backend
fault
latency
last frame
dropped frames
overflow
```

## G7. Implement postprocessing

Recommended:

```text
postprocessing.py
```

Responsibilities:

```text
detector decode
NMS
crop extraction
reader decode
confidence handling
unsupported-domain rejection
ObservationBatch construction
```

Road-law interpretation must remain outside detector/reader postprocessing.

---

# 9. Phase H — temporal semantics

A single frame must not become a current limit.

Implement:

```text
Detection
  ↓
SignTrack
  ↓
temporal consensus
  ↓
stable observation
```

Initial profile to evaluate:

```text
3 unique observations of 5
within about 0.8 s
```

Track:

```text
IoU
family consistency
value consistency
duplicate frame rejection
TTL
lost track
oscillation
```

Do not freeze thresholds as final until evaluation.

---

# 10. Phase I — road ownership and passage

Road ownership states:

```text
OWN
OTHER
UNKNOWN
```

Test:

```text
exit ramp
entry ramp
parallel road
intersection
side road
overhead lane sign
construction
temporary sign
```

Invariant:

```text
correct OTHER-road reading
!=
own-road current limit
```

Passage states:

```text
OBSERVED
AHEAD
CURRENT
UNCERTAIN
UNAVAILABLE
```

A sign disappearing from view is not sufficient proof of passage.

---

# 11. Phase J — publish real `speedVisionState`

Flow:

```text
ObservationBatch
  ↓
tracking/applicability/passage
  ↓
LimitHypothesis
  ↓
speedVisionState
```

Preserve absence explicitly.

A numeric field must never be considered valid when its presence flag is false.

---

# 12. Phase K — finish UI/settings

Keep Vision separate from operational SLA.

UI states:

```text
VISION unavailable
VISION observed
VISION ahead
VISION current
VISION stale
VISION conflict
```

Settings:

```text
Off
Shadow
Observation
Advisory
```

Additional toggles:

```text
Vision warnings
Private diagnostics
```

Default:

```text
Off
```

---

# 13. Phase L — non-interference regression

Compare:

```text
NNSLR OFF
```

against upstream baseline for:

```text
actuator requests
cruise requests
set speed
SpeedLimitResolver output
SLA source
engagement
radar
safety
```

Fault-inject:

```text
Vision missing
Vision malformed
Vision stale
Vision wrong
camera unavailable
backend unavailable
backend crash
invalid model bundle
```

None may affect operational control.

---

# 14. Phase M — replay, benchmark and release progression

Implement:

```text
nnslr replay
```

Replay uses:

```text
recorded camera
rlog/qlog
verified model bundle
runtime configuration
```

Annotations remain evaluation-only.

Then measure on target:

```text
capture→result p50/p95/p99
CPU
accelerator
RAM
temperature
throttling
dropped frames
main-model latency regression
restart behavior
```

Initial performance targets:

```text
p95 <= 150 ms
p99 <= 200 ms
pending queue <= 1
no new primary deadline misses
```

These are targets, not currently verified results.

Release progression:

```text
offline replay
  ↓
source-ready
  ↓
target benchmark
  ↓
SHADOW
  ↓
OBSERVATION
  ↓
ADVISORY
```

Do not jump directly to advisory mode.

---

# 15. Phase N — seal the reapply patchset

After the runtime is stable, update:

```text
integration/sunnypilot/feature.lock.json
```

from:

```text
state = bootstrap
```

to:

```text
state = sealed
```

Record:

```text
source_base_commit
source_feature_commit
logical patch list
core digest
schema binding
model bundle digest
```

Verify:

```text
same-base reapply
new-master reapply
source tests
build tests
protected-surface checks
range-diff/report
```

Do not force-push the known-good branch.

---

# 16. Exact recommended execution order

```text
1. Finish T2 segment-0 investigation
2. Complete 10-point synchronization audit
3. Enrich frame manifests
4. Finish T2 edge-case tests
5. Freeze annotation schema
6. Implement import-annotations
7. Implement validate-dataset
8. Implement build-splits
9. Produce reviewed Swiss dataset
10. Freeze model contract
11. Implement train
12. Train baseline on V100
13. Implement evaluate
14. Mine hard examples
15. Retrain
16. Implement export-onnx
17. Verify export parity
18. Implement package-model
19. Verify bundle
20. Fix Sunnypilot runtime CI
21. Add artifact loader
22. Add runtime preprocessing parity
23. Connect VisionIPC
24. Implement real inference backend
25. Implement postprocessing
26. Implement temporal tracking
27. Implement road ownership
28. Implement passage logic
29. Publish real speedVisionState
30. Finish UI/settings
31. Run non-interference regression
32. Implement replay
33. Benchmark target device
34. Run Shadow
35. Run Observation
36. Enable Advisory
37. Seal reapply patchset
38. Test against newer Sunnypilot master
```

---

# 17. Instructions for coding agents

Every coding agent must:

1. Inspect current Git state before editing.
2. Treat Git as newer than this document when they disagree.
3. Read existing tests before changing behavior.
4. Add regression tests for bug fixes.
5. Never weaken safety/isolation tests to make CI green.
6. Never renumber Cereal/Cap'n Proto bindings automatically.
7. Never use MAP as Vision ground truth.
8. Never route Vision into vehicle control.
9. Never deploy to the comma unless explicitly authorized.
10. Never force-push a stable branch.
11. Never claim completion without fresh test/build evidence.
12. Keep training code out of the Sunnypilot runtime repo.
13. Keep device-specific runtime code out of the pure portable core.

---

# 18. Next major milestone definition

The next major milestone is reached when:

```text
T2 gate closed
canonical annotations implemented
dataset validation implemented
leakage-safe splits implemented
reviewed Swiss dataset available
first baseline model trained
evaluation report available
verified export bundle available
Sunnypilot source CI fully green
```

Only after that milestone should full on-device inference integration become the primary task.

---

# 19. Target architecture

```text
TRAIN_REPO
   │
   ├── data / annotations
   ├── training
   ├── evaluation
   └── export
          │
          ▼
VERIFIED MODEL BUNDLE
          │
          ▼
SUNNYPILOT
 VisionIPC camera
          │
          ▼
 preprocessing
          │
          ▼
 inference backend
          │
          ▼
 ObservationBatch
          │
          ▼
 tracking / ownership / passage
          │
          ▼
 speedVisionState
      ┌───┴────┐
      ▼        ▼
 VISION UI  diagnostics

CAR/MAP ─────► operational SLA
VISION ──────X no operational control
```
