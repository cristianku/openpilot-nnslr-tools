# NNSLR — Remaining implementation and Sunnypilot/Openpilot integration

**Updated:** 2026-09-19  
**TRAIN_REPO:** `cristianku/openpilot-nnslr-tools`  
**RUNTIME_REPO:** `cristianku/sunnypilot`  
**Runtime branch:** `nn-speed-limit-vision`

## 1. Goal

NNSLR must remain an independent, advisory-only vision source:

```text
fcamera / VisionIPC
        ↓
preprocessing
        ↓
detector + reader
        ↓
ObservationBatch
        ↓
temporal tracking / road applicability / passage
        ↓
speedVisionState
        ↓
UI / replay / diagnostics

CAR + MAP → existing SpeedLimitResolver / SLA
VISION   ─X→ no direct control path
```

Permanent invariants:

- never write `CarStateSP.speedLimit`;
- never write or impersonate `LiveMapDataSP`;
- do not add Vision as an operational `SpeedLimitResolver` source in the first release;
- no CAN, acceleration, steering, torque, radar, safety, or cruise-button output;
- do not change the driving model just to make NNSLR work;
- do not initialize Chestnut from a second process;
- map speed may help find annotation candidates, but is never training ground truth;
- runtime inference must be local to the vehicle.

---

## 2. TRAIN_REPO — current status

Baseline before this local T2 continuation:

```text
6cb124f
```

Already implemented:

- pure `speed_vision_core` contracts and validation;
- `FrameRef`, `Detection`, `ObservationBatch`, `RoadContext`, `SignTrack`, `PerceptionHealth`, `LimitHypothesis`, `SourceSnapshot`, `AdvisoryComparison`;
- explicit unknown/unreadable/not-applicable/unavailable states;
- strict speed-value and timestamp validation;
- local Comma route/video/log tooling:
  - `sync-routes` (default SSH host `comma`, all route segments, narrow camera + full rlog)
  - `route-manifest`
  - `inspect-manifest`
  - `video-probe`
  - `extract-frames` (supports a complete `--route` without a shell loop)
  - `log-metadata`
  - `align-route`
  - `alignment-report`
  - `find-candidates`
  - `make-clips`;
- support for local `fcamera.hevc`, `ecamera.hevc`, `qcamera.ts`, `rlog(.zst)`, `qlog(.zst)`;
- EncodeIndex-based video/log alignment.

A real route is now available locally:

```text
00000089--0ac1c0fdec
segments: 0,1,2,3
```

with `fcamera.hevc` and `rlog.zst` for each segment. Other cameras/logs were
removed and the narrow-camera/full-rlog set was downloaded again as requested.

### Immediate TRAIN_REPO work

<!-- [t2-validation] - START -->
1. Route copy and route-wide frame extraction are implemented:
   ```bash
   nnslr sync-routes --route ROUTE
   nnslr extract-frames --route ROUTE
   ```
   Default SSH host: `comma`; `--host` / `NNSLR_COMMA_HOST` override it.
   Data root: `NNSLR_DATA_ROOT`, otherwise `/srv/nnslr-data`.
   The proposed `copy-comma-route.sh` is superseded by this CLI.
<!-- [t2-validation] - END -->

2. Validate the complete T2 pipeline on the real route:
   ```text
   route-manifest
   video-probe
   log-metadata
   align-route
   alignment-report
   find-candidates
   make-clips
   extract-frames
   ```

<!-- [t2-validation] - START -->
   **Current validation:** local end-to-end execution found and fixed parser
   schema loading, HEVC media clock handling, false exact joins and empty HEVC
   clips. Segments 1–3 provide 2,785 aligned full-rate frames; segment 0 has
   1,180 unresolved frames because the logged indices start at 20. Five map
   candidates are projected. The 198 discovery PNGs extracted previously are
   media-only; they are not retroactively capture-aligned by these changes.

   **T2 remains partial.** See [T2_REAL_ROUTE_VALIDATION.md](T2_REAL_ROUTE_VALIDATION.md).
   Resolve/verify the first-segment mapping, perform the ten-clip manual audit,
   and enrich extracted-frame manifests before declaring the gate complete.
<!-- [t2-validation] - END -->

3. Investigate real-world edge cases:
   - HEVC decode behavior;
   - qlog/rlog parser compatibility;
   - unresolved EncodeIndex joins;
   - fcamera/ecamera/qcamera mapping;
   - partial/missing segments;
   - timestamp consistency.

---

## 3. Dataset / annotation work still missing

The following commands are not implemented yet:

```text
nnslr import-annotations
nnslr validate-dataset
nnslr build-splits
```

Required annotation contract should cover at least:

```text
route / segment / frame
bbox
sign family
value state
value_kph optional
linked supplementary panel
own-road / other-road / unresolved
visibility / occlusion
review state
```

Minimum sign vocabulary:

```text
maximum speed
cancellation
zone
variable display
other sign
not a sign
unreadable
```

Hard negatives must include ramps, parallel roads, junctions, advertising, rear of signs, distant signs, glare, blur, rain, night, tunnels, temporary signs, electronic signs, and partial supplementary panels.

Dataset splits must be leakage-safe at route/sequence level rather than randomly mixing adjacent video frames.

---

## 4. Model training still missing

Not implemented:

```text
nnslr train
nnslr evaluate
nnslr mine-hard-examples
```

Recommended model contract:

```text
Detector
   ↓
sign crop
   ↓
Reader / classifier
   ↓
Detection {
  bbox,
  sign_family,
  value_state,
  value_kph,
  detection_score,
  classification_score,
  supported_domain
}
```

The model must recognize evidence; it must not directly output an operational vehicle speed target.

Evaluation must include:

- detector precision/recall/mAP;
- false positives per hour;
- numeric value accuracy;
- sign-family accuracy;
- other-road false positive rate;
- time-to-first-detection;
- stable-value latency;
- difficult confusions such as 30/80, 60/80, 80/100, 100/120.

Hard-example mining must feed errors back into the dataset instead of adding special-case runtime hacks.

---

## 5. Export and model bundle still missing

Not implemented:

```text
nnslr export-onnx
nnslr replay
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
├── model_card.json
├── capability.json
└── sha256sums.json
```

The bundle must freeze:

- input dimensions and dtype;
- color format;
- crop/resize/letterbox behavior;
- normalization;
- tensor layout;
- output names;
- class vocabulary;
- schema/model/training versions;
- hashes.

ONNX should be treated as an interchange/reference format. The final on-device format depends on the verified runtime backend.

---

## 6. Sunnypilot runtime — current source state

Runtime branch:

```text
cristianku/sunnypilot:nn-speed-limit-vision
```

Initial upstream base:

```text
sunnypilot/sunnypilot:master
a5f44653d7f43ad57fef2f546f3916ec4cbf3c56
```

Already present in the runtime branch:

```text
openpilot/sunnypilot/speed_vision/
├── adapters.py
├── runner.py
├── publisher.py
├── lifecycle.py
├── diagnostics.py
├── speedvisiond.py
├── artifacts.lock.json
└── _vendor/speed_vision_core/
```

Current runtime artifact lock explicitly says:

```json
"model": {
  "state": "unavailable",
  "bundle_digest": null
}
```

So the source scaffolding exists, but there is not yet a real NNSLR model connected to the runtime.

Also already introduced:

- dedicated `speedVisionState` messaging;
- separate service registration;
- default-off NNSLR params;
- optional `speedvisiond`;
- UI subscription and separate VISION badge work;
- portable-core snapshot;
- diagnostics scaffolding.

The operational `speed_limit_resolver.py` currently does **not** consume `speedVisionState`. Keep it that way for the advisory release.

---

## 7. Critical Sunnypilot work to actually run the model

### 7.1 Make runtime CI green

A source-level workflow exists:

```text
.github/workflows/nnslr-source.yml
```

The latest observed runtime CI is not green. The remaining failures observed are currently test-path/root issues rather than a demonstrated NNSLR algorithm failure.

Before calling the runtime source-ready:

- fix repo-root resolution in source tests;
- run the full NNSLR runtime test suite;
- validate cereal/schema generation;
- run relevant existing Sunnypilot speed-limit tests;
- run a real source/build check.

### 7.2 Connect `speedvisiond` to VisionIPC

Current daemon behavior is still bootstrap/fail-closed.

Implement:

```text
camerad
  ↓
VisionIpcClient
  ↓
VISION_STREAM_NARROW_ROAD
  ↓
owned frame / verified buffer lifetime
  ↓
NNSLR preprocessing
  ↓
runner
```

Required behaviors:

- connect/reconnect;
- preserve frame ID and SOF/EOF timestamps;
- bounded pending queue;
- discard stale frames;
- track dropped frames;
- invalidate state after producer restart;
- report camera unavailable explicitly.

### 7.3 Implement runtime preprocessing parity

Add a dedicated runtime preprocessing layer that exactly matches the model bundle:

```text
NV12
→ color conversion
→ crop
→ resize/letterbox
→ normalization
→ tensor layout/dtype
```

Training/reference preprocessing and runtime preprocessing must be compared on golden fixtures.

### 7.4 Select and implement the real inference backend

This is still open.

Possible profiles:

**Independent backend**
```text
speedvisiond → NNSLR inference
```

**Accelerator-owner integration**
```text
modeld
 ├─ driving model
 └─ NNSLR auxiliary inference
        ↓
speedVisionObservations
        ↓
speedvisiond
```

Important source constraint already identified: if Chestnut is used, it must remain owned by `modeld`; do not initialize a second Chestnut process.

Before selecting the backend, verify the actual device/runner, operators, memory, latency, thermal headroom, and model format support.

### 7.5 Finish the real runner

The backend should implement:

```python
prepare()
submit_latest(frame)
poll()
health()
close()
```

`prepare()` must verify bundle hashes/config/classes and warm up off-road.

`submit_latest()` must never create an unbounded queue.

`poll()` must produce a real `ObservationBatch` associated with the original capture timestamp.

`health()` must expose availability, faults, latency, last processed capture, dropped frames and overflow.

### 7.6 Implement postprocessing

Needed:

- detector decode;
- NMS;
- crop extraction;
- reader decode;
- confidence handling;
- unsupported-domain rejection.

Do not put road-law interpretation inside the detector/reader.

---

## 8. Temporal semantics still missing/incomplete

A single frame must not become a current speed-limit hypothesis.

Required stages:

```text
Detection
  ↓
SignTrack
  ↓
temporal consensus
  ↓
road ownership
  ↓
passage / activation
  ↓
LimitHypothesis
```

Initial temporal profile to evaluate:

```text
3 unique observations of 5
within roughly 0.8 s
```

but thresholds must be frozen before evaluation.

### Road ownership

Must distinguish:

```text
OWN
OTHER
UNKNOWN
```

A correctly read sign on an exit ramp must not become the current limit of the main carriageway.

### Passage

States should remain explicit:

```text
OBSERVED
AHEAD
CURRENT
UNCERTAIN
UNAVAILABLE
```

A sign disappearing from camera view is not, by itself, proof that the vehicle passed it.

---

## 9. CAR / MAP / VISION comparison

The comparator is advisory only.

```text
CAR
MAP
VISION
 ↓
agree / conflict / insufficient_data
```

Do not:

- boost Vision confidence just because MAP agrees;
- fill a missing Vision value from MAP;
- overwrite Vision with CAR;
- route Vision into the operational SLA resolver.

---

## 10. UI and settings still to finish

A separate VISION HUD element has been started.

Remaining:

- real-device layout validation;
- metric/imperial display;
- observed/ahead/current/unavailable/stale/conflict rendering;
- shadow mode hidden from on-road UI;
- Settings UI for:
  ```text
  Off
  Shadow
  Observation
  Advisory
  ```
- warning toggle;
- diagnostics toggle;
- snapshot/layout tests.

The first visible release should show a clearly separate source, e.g.:

```text
VISION 80
```

It must not silently replace the operational SLA/MAX badge.

---

## 11. Privacy / diagnostics remaining work

Audit these Sunnypilot paths before enabling real diagnostics:

```text
loggerd
uploader
sunnylink
crash logs
backup
file sharing
```

Default behavior:

- no raw frames;
- no route upload;
- no GPS dump;
- bounded private diagnostics;
- diagnostics failure must not block the driving system.

---

## 12. Source and device validation

Before any on-road use, verify non-interference against the newly selected upstream baseline.

At minimum compare:

```text
actuator requests
cruise requests
set speed
SpeedLimitResolver output
SLA source
engagement behavior
radar behavior
safety configuration
```

Inject:

```text
Vision absent
Vision valid
Vision incorrect
Vision stale
Vision malformed
backend unavailable
backend crash
camera unavailable
```

None may affect operational control.

Target benchmark later:

- capture-to-result p95 ≤ 150 ms;
- capture-to-result p99 ≤ 200 ms;
- pending queue ≤ 1;
- no new primary deadline misses;
- measure RAM, temperature, throttling, dropped frames and main-model latency regression.

---

## 13. Replay and release progression

Recommended progression:

```text
offline replay
  ↓
source-ready
  ↓
target benchmark
  ↓
shadow
  ↓
observation UI
  ↓
advisory UI
```

Replay must use recorded camera/log data as runtime input; annotations remain evaluation-only and must never act as oracle runtime inputs.

---

## 14. Reapply-on-new-master tooling

A dedicated TRAIN_REPO branch exists:

```text
nnslr/sunnypilot-reapply-v1
```

It already contains the reapply design/tooling, including:

```text
integration/sunnypilot/feature.lock.json
integration/sunnypilot/hook-map.json
integration/sunnypilot/protected-surfaces.json
integration/sunnypilot/schema-bindings.json
scripts/nnslr_reapply.py
.github/skills/nnslr-reapply/SKILL.md
```

The latest observed CI on that tooling branch is green.

However, the feature lock is still:

```text
state = bootstrap
source_feature_commit = null
patches = []
```

Therefore the runtime feature is **not yet sealed as a reproducible patchset**.

After the runtime stabilizes:

1. identify the exact known-good runtime commit;
2. organize the NNSLR commits into a clean logical patch series;
3. change the feature lock to `sealed`;
4. record source base, feature commit, patch list, core/model/schema hashes;
5. reapply to the same base and compare the resulting tree;
6. reapply to a newer real Sunnypilot master;
7. run compatibility/build/source tests;
8. generate diff/range-diff/report;
9. keep the previous known-good version intact.

Final maintenance workflow:

```text
current Sunnypilot master
        ↓
nnslr-reapply skill/tooling
        ↓
new candidate branch
        ├─ source tests
        ├─ build
        ├─ compatibility report
        └─ no automatic deployment
```

---

## 15. Recommended implementation order from here

### A — Real data
<!-- [t2-validation] - START -->
- [x] `sync-routes` with default host `comma`, resume and segment discovery;
- [x] route-wide `extract-frames`;
- [x] run the local T2 commands on the four-segment route and fix observed failures;
- [x] project five map candidates and generate review clips;
- [ ] establish a verified mapping for the nonzero first-segment index;
- [ ] complete the ten-clip manual alignment audit and timestamp/geometry frame manifests;
- [ ] extend validation to missing/truncated recordings and other camera streams.
<!-- [t2-validation] - END -->

### B — Dataset
- annotation schema;
- annotation import/review;
- negative/challenge examples;
- dataset validation;
- leakage-safe splits.

### C — Model
- detector baseline;
- reader baseline;
- train on V100;
- evaluate;
- hard-example mining;
- retrain/finalize.

### D — Export
- ONNX/reference export;
- parity tests;
- immutable bundle;
- hashes/capabilities.

### E — Sunnypilot runtime
- fix runtime CI;
- VisionIPC input;
- preprocessing parity;
- real inference backend;
- postprocessing;
- temporal tracking;
- ownership/passage;
- UI/settings/privacy.

### F — Device
- build;
- off-road smoke test;
- runtime benchmark;
- fault tests;
- shadow;
- observation;
- advisory.

### G — Maintenance
- seal patchset;
- same-base reproducibility test;
- newer-master reapply test;
- rollback documentation.

---

## 16. Definition of Done

The project is not complete until all of the following are established:

**Training**
- versioned dataset and leakage-safe split;
- evaluated detector/reader;
- hard negatives;
- final model identified by hash;
- export parity.

**Runtime**
- real camera input;
- exact preprocessing;
- verified target backend;
- bounded queue/fault handling;
- real `speedVisionState`;
- temporal state;
- ownership and passage;
- separate UI.

**Safety**
- operational CAR/MAP resolver unchanged;
- no CAN/control output;
- default off;
- fail-closed;
- regression/non-interference suite green.

**Target**
- build passes;
- benchmark passes;
- shadow evidence collected;
- thermal/memory behavior acceptable.

**Maintenance**
- sealed patchset;
- working reapply skill/tool;
- reapply tested against a newer master;
- old version preserved;
- rollback documented.

## 17. Immediate next checkpoint

The next useful milestone is:

```text
real route
  → deterministic aligned dataset inputs
  → first annotated dataset
  → first trained detector/reader
  → verified model bundle
  → Sunnypilot VisionIPC inference
```

Until a real model bundle exists, the Sunnypilot runtime should remain fail-closed and report Vision as unavailable rather than fabricating a speed limit.

<!-- [preannotation] - START -->
### Pretrained annotation assistance

`nnslr preannotate --route ID` and `nnslr review --route ID` now provide
CPU-only model proposals and browser review with the standard data-root
default. Sign detection uses a pinned Vietnam YOLO ONNX export. Asphalt OCR
uses PP-OCRv4 and a pinned Cityscapes road-segmentation model: numeric text
must pass an independent road-surface/context check before becoming a
`road_marking_candidate`. It is never automatically an applicable limit.

The UI supports box addition/removal, class/value corrections, explicit frame
confirmation, browser persistence and export of confirmed frames with original
proposal provenance. Unreviewed and empty model outputs are not ground truth.

Local CPU validation (2026-09-19): all 198 extracted route frames completed,
with 100 proposals across 69 frames. Four proposals were road-number candidates
(all 30), including the requested segment-0 frame 27 and two following frames.
The additional road proposal was visually checked on a separate zone-30 marking.
This is successful execution and a small positive check, not an accuracy metric.
Synthetic perfect-score OCR readings placed in actual wall/advertisement/parked
vehicle mask regions were rejected; the asphalt region was accepted.

The regression run passed 165 tests, including optional real Chrome review/export
and storage-failure checks; seven optional log-parser tests were skipped because
their separate environment was not configured. The core selftest passed all 18
checks. A real-image browser check also exercised box drawing/removal, value
correction, confirmation reset, reload persistence and export. No container or
comma installation was performed for this extension.

Still outstanding: curated positive/negative validation across scenes,
Swiss sign-domain assessment, sequence consistency and road applicability,
review export import into the canonical annotation schema, source-frame/time
alignment, dataset validation/splits and actual training. The OCR/segmentation
scores do not establish these properties. See README for commands, model
provenance and the external weights' license constraints.
<!-- [preannotation] - END -->
