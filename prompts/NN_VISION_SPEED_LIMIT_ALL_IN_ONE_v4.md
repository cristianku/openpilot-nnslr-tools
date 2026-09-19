# NN Vision Speed Limit — All-in-One Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development`, when available, to execute one approved increment at a time. Checkboxes track work; they do not grant permission to deploy to a vehicle, change its software, or start a GPU workload.

**Goal:** Build a locally trainable Swiss speed-sign recognizer for Cristian’s sunnypilot fork, first as an offline replay tool, then as a measured shadow service, and finally as an isolated advisory display.

**Architecture:** Two repositories separate model development from vehicle integration. The proposed `cristianku/openpilot-nnslr-tools` owns dataset preparation, training, evaluation, export, documentation and a hardware-independent reference core; `cristianku/openpilot` owns only the audited device adapters, inference scheduling, messaging and advisory UI. Versioned, checksum-verified artifacts join them. A compact detector and sign reader produce timestamped observations. Independent temporal and road-context modules decide what was seen, which road it belongs to, and whether its application point has been passed. An advisory consumer compares Vision with Car and Map without changing the operational speed-limit resolver or vehicle-control inputs.

**Tech stack:** Python for data tools, temporal logic and integration; PyTorch on Tesla V100 for training; ONNX for a portable reference artifact; the target fork’s pinned tinygrad only when that runtime has been demonstrated compatible. Reuse the existing UI and messaging infrastructure after auditing the active implementation; no new web application or general-purpose vision-language model.

**Specification:** This file contains the project charter, design, data contracts, implementation backlog, tests, acceptance gates, release requirements, references and initial Codex prompt. Sections 2–12 are the technical specification; section 14 contains the 13 execution tasks. No earlier plan or separate design document is required.

**Document:** `NN_VISION_SPEED_LIMIT_ALL_IN_ONE_v4.md` · Revision 4: standalone training repository · 17 September 2026.

**Evidence status:** The openpilot/runtime and framework observations in sections 2 and 17 [S1–S19] are carried forward from the supplied plans and were not independently re-fetched in this revision. The README of `cristianku/openpilot-nnlc-tools` was read through the GitHub connector on 17 September 2026 to verify the organizational reference [S20]. The new repository name and layout are proposals, not existing resources or tested implementations. T0 must check the actual checkout and, where authorized, the target environment. A recorded source fact, a design proposal and a measured result are different categories.

**Execution status:** Planning-document update and read-only inspection of the NNLC reference README only. No GitHub repository was created, no files were pushed, and no training scripts or project packages were implemented. No device software, training, inference, deployment or hardware benchmark was executed. All performance numbers remain proposed experiment settings or acceptance targets. Task paths, CLI commands and script names are implementation requirements, not claims that those files already exist.

**Revision 4 change:** The earlier file put training tools under the openpilot tree; this revision replaces that layout with an explicit standalone training repository, a shared-core/artifact boundary, repository-specific tests and an updated Codex handoff. The previous file is preserved separately, but is not required to execute this specification.

## Contents

- [Global constraints](#global-constraints)
- [1. Project charter and binding decisions](#section-01)
- [2. Recorded source baseline and verification checklist](#section-02)
- [3. Scope, capabilities and release sequence](#section-03)
- [4. Architecture and boundaries](#section-04)
  - [4.3 Standalone GitHub training repository](#training-repository)
- [5. Dataset, annotation and reproducibility](#section-05)
- [6. Training, export and model selection](#section-06)
- [7. Runtime data contracts](#section-07)
- [8. Temporal logic, road ownership and passage](#section-08)
- [9. Car / Map / Vision comparison and UI](#section-09)
- [10. Device runtime, Chestnut and failure behavior](#section-10)
- [11. Evaluation and release gates](#section-11)
- [12. Repository file maps, scripts and test boundaries](#section-12)
- [13. Dependency graph and approval gates](#section-13)
- [14. Task-by-task execution backlog](#section-14)
- [15. Effort, decision records and risk ownership](#section-15)
- [16. Codex handoff prompt](#section-16)
- [17. Sources and evidence provenance](#section-17)
- [Appendix A. Configuration and threshold register](#appendix-a)
- [Appendix B. Deliverables, approval record and release checklist](#appendix-b)

---

<a id="global-constraints"></a>

## Global constraints

- Do not alter CAN, radar, ACC, steering, safety code, torque tables, acceleration limits, or driving-model weights.
- Keep Vision out of the control path throughout this plan, including its final advisory increment.
- Default to disabled. Never enable Vision automatically because hardware or a model file is present.
- Do not change the user’s existing SLA modes, confirmation behavior, offsets or saved policies.
- Keep training in a distinct GitHub repository, proposed as `cristianku/openpilot-nnslr-tools`, not a folder or branch inside the driving fork and not a fork of NNLC. Its final name, visibility and creation/publication require owner approval.
- Use a separate local workspace/branch for the training repository. Develop runtime integration in an isolated openpilot worktree, proposed branch `feat/vision-speed-limit-shadow`; never switch or update the checkout used for driving without separate permission.
- Do not install the training package, PyTorch training dependencies, optimizer checkpoints or private datasets on the comma. Runtime imports use a pinned minimal core snapshot and compatible model bundle; no import or download from a moving training branch.
- Use local inference on the vehicle. No dependence on the home server, mobile connectivity or a cloud service while driving.
- Do not replace host GPU drivers, upgrade the comma’s global dependencies or interrupt existing LLM services to make this project work.
- Preserve `unknown`, `unreadable`, `not_applicable` and `unavailable` as different outcomes. None means zero speed or unrestricted speed.
- Separate experimental logs from existing upload pipelines. No automatic publication of routes, images, coordinates, model-training data or diagnostic payloads.
- Automatic speed control using Vision is a separate project, with a separate design and approval. It is not an additional toggle at the end of this plan.

---

<a id="section-01"></a>

## 1. Project charter and binding decisions

**First useful result:** a reproducible replay of selected Swiss routes with sign boxes, recognized values, timestamps, rejection reasons and a read-only comparison with the car/map estimates. An observed number must not be presented as the current applicable limit unless road ownership, conditions, passage and continuity have also been established.

| Decision | Binding implementation requirement |
|---|---|
| Independent sign-recognition project | Keep this separate from lateral-model refinement, NNLC, PSA radar/longitudinal changes and driving-model training. |
| Standalone training repository | Proposed name `cristianku/openpilot-nnslr-tools`; own all training instructions, executable pipeline scripts, architecture definitions, configurations, tests and export tooling here. A separate GitHub repository is required; a nested openpilot training folder is not an equivalent implementation. |
| Two-repository contract | Training produces immutable model/core artifacts. The runtime fork consumes pinned artifacts and contains only integration-specific code; no training dependency or automatic model update in the vehicle. |
| Local training; local vehicle inference | Use the available V100 environment for approved training. Vehicle operation must not depend on that server or an internet connection. |
| Audit the selected runner | Identify the actual Python/native runner, driving-model bundle, source revision and hardware before selecting a device integration path. |
| Separate advisory path | Leave operational Car/Map resolution, Assist, confirmation, offsets and set-speed behavior unchanged under every mode. |
| Four distinct interpretation problems | Evaluate detection/value reading, road relevance, passage and limit continuity separately, then end to end. |
| V100 compatibility gate | Pin an environment only after checking supported GPU architectures and executing a real forward/backward step. |
| Leakage-resistant evaluation | Group by physical sign sites, encounters and related routes; freeze the holdout before model/threshold selection. |
| Bounded runtime and private evidence | Limit queues, snapshots, crop work and logging; reject stale data; audit upload/backup exposure. |
| Evidence-based progression | Complete tasks T0–T12 through gates G0–G7. Observation-only remains a valid deliverable when interpretation or device gates are not established. |

**Non-goal:** this plan does not authorize Vision-driven acceleration, braking or automatic cruise-speed changes. A future operational Vision source requires a separate specification and review; it is not an unfinished step in this plan.

**Working rule:** preserve source provenance and uncertainty. Do not label car/map output as camera output, turn ambiguity into a default number, or promote a curated successful demo into a runtime/reliability claim.

---

<a id="section-02"></a>

## 2. Recorded source baseline and verification checklist

### 2.1 Recorded repository snapshot

**Repository:** `cristianku/openpilot`  
**Branch recorded in the supplied plans:** `psa-torque-sunny-testing`  
**Recorded reference commit:** `4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9`. The supplied refined plan records its source reads against this commit. This is a historical reference, not proof of the device’s installed revision. Confirm it in T0. [S1]

All paths below are relative to the repository root as recorded in the supplied source snapshot. Retain the leading `openpilot/` for that layout; if T0 establishes a different checkout layout, update the path map and test commands together rather than creating a duplicate package tree. The table describes the recorded snapshot, not a new inspection during consolidation.

| File in the recorded snapshot | Recorded observation to check in T0 | Consequence for this project |
|---|---|---|
| `openpilot/selfdrive/modeld/modeld.py` | `ChestnutState` contains an explicit single-owner comment. The runtime also copies camera buffers and synchronizes model output. | No second process independently initializing Chestnut; no assumption that an existing callback is a safe scheduling slot. [S2] |
| `openpilot/system/manager/process_config.py` | Selects Python `modeld` for the stock runner and native `modeld_tinygrad` for the tinygrad runner. | Record the actual selected bundle and process before designing an inference hook. Initially support only the runner actually validated. [S3] |
| `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py` | Reads `carStateSP.speedLimit` and `liveMapDataSP`; maps saved policies to Car and Map. Its existing combined policy selects the lower positive solution. | Preserve these policies exactly. Do not insert Vision into the combined policy or counterfeit `carStateSP`. [S4] |
| `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py` | Contains Assist state and target-speed logic. | Treat this as a protected control-path boundary, not the place to add an experimental reader. [S5] |
| `openpilot/cereal/custom.capnp` | Operational speed-limit source enum is `none @0`, `car @1`, `map @2`. | Keep these values and their meanings unchanged. Vision gets its own advisory schema. [S6] |
| `openpilot/cereal/services.py` | Has narrow/wide road camera and encode-index services; service definitions specify logging and decimation. | Use the actual stream names and audit logging explicitly. Do not assume older `roadCameraState` names. [S7] |
| `openpilot/tools/lib/route.py` | Defines `fcamera.hevc`, `ecamera.hevc`, `qcamera.ts`, `rlog` and `qlog` as separate route files. | Video is required for training; logs supply metadata and state. Prefer the full relevant log over a decimated substitute. [S8] |
| `openpilot/selfdrive/ui/ui.py` | The UI entry point is Python and selects different layouts. | Reuse its state/subscription and layout conventions after tracing the active widget path. [S9] |
| `.gitmodules` | Uses `opendbc_repo`, `msgq_repo` and `tinygrad_repo`, among other submodules. | Record their exact revisions; do not silently update them. [S10] |

The public SLA documentation also distinguishes Information, Warning and Assist. That distinction supports the rollout concept, but the pinned fork’s actual consumers remain the authority for integration. [S11]

### 2.2 Not established by reading GitHub

The following must become evidence records during execution; this document does not assert their values:

- The commit and local modifications currently installed on `comma`.
- Device model, active cameras, selected driving-model bundle and running model process.
- Whether Chestnut is attached, its GPU, usable memory and available compute margin.
- Whether one or two V100s are available for this workload, their current allocation and free memory.
- Available full-resolution routes, complete encode indexes and calibration data.
- Achievable detector throughput, capture-to-result latency or thermal stability.
- Final target-runtime operator support for the chosen network.
- The exact UI widget, parameter registry and uploader paths that need integration changes.

An unverified hardware capability keeps that backend disabled. It does not block the offline deliverable.

### 2.3 Audit watchpoint: time domains

The earlier source review flagged resolver arithmetic involving a GPS field named `unixTimestampMillis` and `time.monotonic()`. This is a reason to trace the producer and establish the actual field semantics; it is **not sufficient evidence on its own to diagnose a bug**. Do not copy that arithmetic into Vision, and do not refactor the existing resolver in this feature branch. Record any confirmed unrelated issue separately. [S4]

---

<a id="section-03"></a>

## 3. Scope, capabilities and release sequence

### 3.1 Four separately approved increments

| Increment | User-visible result | Vehicle-control influence |
|---|---|---|
| **A — Offline perception** | Annotated videos, detections, read values, rejection reasons and model benchmark. | None; runs on the workstation/server. |
| **B — Offline interpretation** | Tracks, road applicability, passage estimates, sign-event timeline and comparison with Car/Map. | None. |
| **C — Device shadow** | Private diagnostic output and performance report. No on-road badge or warnings. | None, with semantic and runtime non-interference tests. |
| **D — Advisory UI** | Clearly labeled observed/ahead/current hypotheses and optional separately reviewed warnings. | None; operational SLA remains on its existing Car/Map path. |

A useful result can ship at each level. In particular, uncertain passage or road attribution need not stop sign-observation research, but must prevent a result being advertised as the current limit.

### 3.2 Initial supported domain

Start with **daytime, adequately visible, static numeric maximum-speed signs in Swiss road scenes**, with no unresolved supplementary panel. Treat motorway approaches, exit ramps and gantries as first-class evaluation categories, not as an afterthought.

The first training class vocabulary may include `20, 30, 40, 50, 60, 70, 80, 100, 120`, plus explicit unknown/unreadable and non-speed classes. This is a proposed model vocabulary, not a claim about every permitted Swiss sign value. Publish support only for classes that actually have suitable training and independent test coverage.

Detect or label the following from the beginning, but default to abstention for current-limit interpretation until separately implemented and verified:

- Variable LED displays and lane-specific gantry signs.
- Zone entry/exit, general-limit signs and cancellation signs.
- Time, weather, vehicle-class, distance or directional supplementary panels.
- Night, severe glare, snow covering signs, illegible signs and strong motion blur.
- Opposite carriageways, parallel roads, ramps, vehicle-mounted signs and non-road numbers.
- Foreign units or a country/context transition that has not been resolved.

Do not infer that a panel is unconditional because its text was too small to read. A crop that excludes the panel is not evidence of its absence.

### 3.3 Meaning of the display states

**Observed:** a sign was read, but road relevance or current applicability is unresolved.  
**Ahead:** a relevant sign is confirmed but its application point has not been passed.  
**Current hypothesis:** road relevance, conditions and passage have all passed their respective gates; continuity remains valid.  
**Unavailable:** the service, input, runtime or context is no longer usable.  
**Unknown:** the system is functioning but cannot establish a value.

Avoid the unqualified label “legal speed limit.” This subsystem is an advisory estimate, not an authoritative determination.

<a id="section-04"></a>

## 4. Architecture and boundaries

```text
OFFLINE                                      ON DEVICE, AFTER APPROVAL
Local route video + logs                     Road camera frames + read-only state
          |                                                |
    Frame/time adapter                              Frame/time adapter
          |                                                |
    Detector -> crop reader                   Validated inference backend
          |                                                |
          +-------------- ObservationBatch ----------------+
                                 |
                         Tracker / sign reader consensus
                                 |
                         Road applicability estimator
                                 |
                         Passage / continuity estimator
                                 |
                       Experimental limit memory
                                 |
                    speedVisionState advisory message
                       /                       \
              private diagnostics       advisory comparison + UI
                                                  ^
                                  read-only snapshots of Car / Map

Existing Car / Map -> existing SpeedLimitResolver -> Assist -> vehicle control
                         UNCHANGED; no Vision input in this plan
```

### 4.1 Component responsibilities

| Component | Owns | Must not own |
|---|---|---|
| Frame adapter | Camera identity, pixel layout, timestamps, capture continuity. | Sign interpretation or map policy. |
| Perception backend | Preprocessing, detection, sign-family/value outputs and measured inference cost. | Persistent speed state or GPU selection behind the caller’s back. |
| Tracker | Same-object association and evidence accumulation. | Legal rules or claiming that a sign belongs to the ego road. |
| Applicability estimator | Ego-road/lane attribution with reasons and uncertainty. | Reading the value from a map and reporting it as camera evidence. |
| Passage estimator | Ahead/passed/unknown and an uncertainty interval. | Treating disappearance from the image as proof of passage. |
| Limit memory | Current advisory hypothesis and explicit invalidation. | Control targets, offsets or cruise-button requests. |
| Advisory comparator | Separate Car/Map/Vision evidence and conflicts. | Mutating the existing resolver or silently changing saved source priority. |
| UI | Labeled information and visibility rules. | Resolving ambiguity by choosing a more convenient value. |

### 4.2 Choice of model architecture

**Preferred baseline:** one compact sign-family detector followed by a small crop classifier. It exposes whether an error came from finding the sign, recognizing its family or reading its value. The crop must retain the sign border and enough surrounding context to associate supplementary panels.

**Comparator:** a single-stage detector with value-specific classes. It can remove a crop-classification pass, but still requires the same road-context and temporal modules.

Use YOLOX-Nano and YOLOX-Tiny as reproducible baselines, not as a declaration of the best contemporary architecture. The authors provide lightweight variants and ONNX deployment tooling. Code, pretrained weights and training data still need individual license checks before redistribution. [S12]

A general VLM, an OCR-only pipeline, and retraining the driving model are not part of this increment. OCR may become an auxiliary experiment for supplementary text, never a shortcut around sign-family validation.


<a id="training-repository"></a>

### 4.3 Standalone GitHub training repository

#### 4.3.1 Name, purpose and reference

**Recommended repository:** `cristianku/openpilot-nnslr-tools`  
**Project name:** openpilot NNSLR Training Tools  
**NNSLR:** Neural Network Speed Limit Recognition  
**Naming status:** recommendation pending owner confirmation. No repository has been created or name reserved by this document.

Suggested GitHub description:

> Reproducible data preparation, training, evaluation and export of an independent neural network for camera-based speed-limit sign recognition in openpilot and sunnypilot.

The naming and complete-workflow approach follow the user's existing `cristianku/openpilot-nnlc-tools`. Its README describes Python data utilities and a Julia NNLC training pipeline [S20]. Reuse the idea of clear setup instructions, scripts and reproducible stages—not lateral-control labels, architecture, hyperparameters or deployment assumptions. NNSLR is a new vision project; it is not a lateral-torque network or a replacement for the main driving model.

The core recognizer should not depend on a Peugeot model, a particular CAN message or a Swiss-only repository name. Keep Swiss annotation profiles and experimental supported-domain settings in configuration; do not claim support for other countries before evaluating them.

#### 4.3.2 Ownership and dependency direction

| Asset or responsibility | Canonical owner | Boundary |
|---|---|---|
| All contributor, setup, annotation, training, evaluation and export instructions | Training repository | A fresh clone contains the complete written workflow; no dependence on this conversation. |
| Data inventory, optional explicit downloads, extraction, annotation import/validation and splits | Training repository | Original videos/logs and private labels stay in a separate local data root; normal training needs no device access. |
| Detector/reader architecture, losses, augmentations, training loop and experiment configs | Training repository | No vehicle-control imports, host-driver changes or implicit GPU use. |
| Portable reference types, preprocessing, tracking, applicability, passage, state and advisory comparison | `src/speed_vision_core/` in training repository | Hardware-independent reference; no PyTorch, tinygrad, VisionIPC or messaging import. It is also used by offline replay. |
| ONNX export, numerical/semantic parity, packaging and model cards | Training repository | Bundle includes all detector/reader components and a versioned contract. Runtime compatibility is measured separately. |
| Live camera/vehicle adapters, selected-backend scheduling and process lifecycle | Runtime repository, `cristianku/openpilot` | Only these modules read live camera/messaging context. Shared Chestnut ownership remains governed by section 10. |
| Advisory schema, UI, controls-isolation tests and bench regressions | Runtime repository | No Vision input to the operational resolver or Assist. |
| Original data, derived images, full run checkpoints and private reports | Local storage outside both repositories | Back up separately; do not publish through Git or automatic CI artifact uploads. |

**Independence requirement:** installing the tools and training/evaluating an image dataset must not require cloning or building the complete openpilot runtime. A separately pinned optional adapter may decode authorized local openpilot routes; the baseline also accepts ordinary image files plus the portable manifest. Importing a general training module must not import that adapter or initialize a connection to a vehicle.

**Shared-core choice:** use two repositories, not a third infrastructure project. Keep the pure reference core canonical in the training repository. For device integration, export a small source snapshot with its tests, license and hashes; vendor only that snapshot under the runtime package's `_vendor/speed_vision_core/`. Review and pin it in a runtime commit. Do not hand-maintain two implementations of the state machine or make the vehicle import a development checkout. Any backend-specific preprocessing implementation must pass the same golden-input/coordinate tests against the reference.

The core snapshot must pass an import test in an environment without the training framework. Its allowed dependencies must be explicitly recorded and supported in the audited target environment. Do not silently introduce a target package upgrade to satisfy them.

#### 4.3.3 Artifact promotion, not automatic deployment

```text
openpilot-nnslr-tools                  private local storage
  code + docs + configs  <----------  images / video / labels / split manifests
            |
      approved training on selected V100
            |
      reference checkpoint + evaluation evidence
            |
      export and validate
            |
      local versioned model bundle + minimal core snapshot
            |
      explicit review / compatibility check / artifact selection
            v
cristianku/openpilot
  pinned artifact lock + reviewed core snapshot + live adapters
            |
      authorized bench -> shadow -> advisory UI
```

Code versions, model-bundle versions and dataset versions are separate identities. For each candidate record training commit, reference-core tree hash, dataset/split hashes, full config and environment hashes, seed, component hashes, contract version and evaluation capabilities. The runtime additionally records its own commit, selected backend, device/model-runner support matrix and the approved bundle hash.

Use local artifact storage first. Publishing code or release assets to GitHub is a separate owner-approved operation after license/privacy checks. Do not commit large training checkpoints, raw datasets, credentials or private route evidence. A release directory or candidate tag is not evidence of vehicle validation.

No device-side `git pull`, `pip install`, moving `latest` download or automatic promotion after a successful training run. Runtime artifact selection is explicit, checksum-verified, contract-checked and tested on the bench before any shadow session. Missing or mismatched core/model contracts leave Vision unavailable; they must not affect the existing control system.

#### 4.3.4 Repository-level definition of done

- [ ] Owner confirms the repository name and visibility; any remote creation/push has separate recorded authorization.
- [ ] A clean local checkout installs in an isolated environment and runs core/data/CLI tests without a GPU, a comma device, private data or a full openpilot checkout.
- [ ] `README.md` gives a CPU/synthetic quickstart and a distinct, explicitly authorized real-data/V100 workflow; each documented command is exercised in the appropriate test environment.
- [ ] Versioned scripts cover data acquisition where requested, preparation, annotation import, validation, splits, training/resume, evaluation, hard-example mining, export, replay and bundle verification.
- [ ] All train/evaluate/export functions are executable scripts or CLI commands, not notebook-only steps. Notebooks may be optional diagnostics only.
- [ ] A synthetic fixture proves pipeline wiring. A separate approved V100 run proves actual training/export behavior; neither is reported as sign-recognition accuracy on roads.
- [ ] The model card, manifest, component hashes and minimal core snapshot can be consumed by the runtime adapter without installing the training environment.
- [ ] CI checks code/docs/fixtures without exposing private data or triggering vehicle deployment. GPU runs are explicit trusted jobs, not automatic execution of untrusted pull requests on the home server.


---

<a id="section-05"></a>

## 5. Dataset, annotation and reproducibility

### 5.1 Local storage contract

Choose a dedicated local dataset root after checking disk space. The following is the **proposed structure**, not an assertion about existing files:

```text
speed-vision-data/
  raw/                         # Selected original video/log files; immutable
  manifests/routes.jsonl       # File identity, availability, hashes, gaps
  manifests/frames.jsonl       # Frame-to-capture mapping
  annotations/objects.jsonl    # Sign boxes, values, panels and associations
  annotations/events.jsonl     # Encounter-level truth and passage intervals
  splits/v001.json             # Frozen group assignments and rationale
  derived/frames/              # Regenerable extraction cache
  derived/crops/               # Regenerable crop cache
  runs/                        # Config, dataset hash, weights and training reports
  evaluation/                  # Frozen test reports; never input to training
  private/location-index.json  # Restricted mapping of anonymized site identifiers
```

Use content hashes for original files, annotations and split definitions. Do not hash an entire video once per frame: store its hash once in the file manifest and reference it from each frame record. Never commit raw data, coordinates or machine-specific absolute paths to either code repository. Configure this directory with `NNSLR_DATA_ROOT`; keep it outside both repository roots. Version sanitized schemas, synthetic fixtures and public provenance summaries in Git; private manifests and location indexes remain local. Dataset identity is tracked by hashes without requiring publication of its contents.

### 5.2 Frame manifest fields

| Field | Requirement |
|---|---|
| `schema_version` | Integer; reject unsupported versions. |
| `route_id`, `session_id`, `segment_index` | Preserve route order and a monotonic-clock/session identity. |
| `stream` | Canonical `narrow_road` or `wide_road`, mapped explicitly to the fork’s services. |
| `video_relpath`, `video_sha256` | Relative to dataset root; reject traversal or remote URLs. |
| `decoded_frame_index`, `encoded_frame_id` | Different concepts; either may be absent if the available metadata cannot establish it. |
| `capture_mono_ns`, `capture_reference` | Timestamp plus whether it represents SOF, EOF or another verified source. |
| `alignment_status` | `exact`, `bounded_estimate`, `unresolved`; retain the estimated error when applicable. |
| `width`, `height`, `pixel_format`, `color_metadata` | Required for reproducible preprocessing. |
| `calibration_id` | Explicitly absent when calibration was not recorded or validated. |
| `route_group`, `site_group`, `encounter_id` | Used for split isolation and event metrics. |
| `frame_sha256` | Optional hash of a materialized canonical image, not a substitute for provenance. |

Rules: no silent remapping of duplicate frame IDs, no assumption that frame N is log message N, and no interpolation across a missing segment. Preserve genuine gaps. A missing exact timestamp can still permit crop classification, but excludes that sample from precise passage/latency claims.

### 5.3 Alignment procedure

1. Inventory files and complete/missing segments before decoding anything.
2. Read the actual narrow/wide encode-index records and camera-state records available in that pinned version. Establish what each index and timestamp refers to.
3. Join records using documented frame/segment identifiers; check counts and monotonicity per stream and session.
4. Decode in presentation order, preserving the encoder-to-capture relationship. Handle partial GOPs and decoder errors explicitly.
5. Produce an alignment report: unmatched frames, unmatched metadata, duplicates, discontinuities and estimated error bounds.
6. Visually inspect at least ten short clips with timestamps overlaid, including a segment boundary and a known sign passage.

Prefer full video plus the relevant `rlog`. A `qlog` may contain useful metadata but must not be assumed to have every state sample needed. The fork’s service registry explicitly includes decimation settings. [S7–S8]

### 5.4 Annotation vocabulary

Every sign object needs a bounding box in **original-image pixel coordinates**, a track/encounter identifier, sign family, readable value or an explicit unknown, visibility attributes, and linked supplementary-panel boxes.

Maintain independent labels for:

- `family`: maximum speed, cancellation, zone, variable display, other sign, not a sign, unreadable family.
- `value_kph`: integer only when legible; do not complete a partly visible number from the map.
- `applicability_truth`: own road, other road, ambiguous; optional lane scope.
- `condition_truth`: no panel, panel resolved, panel unresolved, unknown panel presence.
- `passage_truth`: earliest and latest plausible crossing times, or unobservable.
- `label_basis`: directly visible evidence, additional local context or adjudicated reviewer decision.
- `quality`: size in native pixels, occlusion, blur, glare and truncation.

**Critical separation:** annotation truth must live in evaluation/annotation inputs, never in runtime input objects. An offline run using human-supplied road ownership or passage is an **oracle diagnostic**, not an end-to-end result.

### 5.5 Pilot collection and coverage

Start with approximately **2,000–5,000 annotated frames from 300–500 encounters**, where available. These are collection budgets, not evidence of deployment reliability. Count distinct physical sites separately from repeated visits.

A pilot coverage target is:

| Scene category | Initial sampling target | Main question |
|---|---:|---|
| Clear relevant roadside signs | 120 encounters | Does detection and number reading work? |
| Motorway approaches and gantries | 80 encounters | Are small, distant and overhead signs retained? |
| Exit ramps, parallel roads, adjacent/opposing lanes | 80 encounters | Can the system avoid the wrong road? |
| Temporary signs and supplementary panels | 40 encounters | Does it abstain rather than drop conditions? |
| Cancellation, zones, LED and other excluded semantics | 40 encounters | Are unsupported events rejected or surfaced as unknown? |
| Sign-free driving and misleading numbers | At least 60 minutes initially | How often are false events produced outside selected sign clips? |

Targets can overlap but must be reported without double-counting the encounter total. Collect safely during ordinary trips; no interaction with annotation tools while driving.

Sampling plan: roughly 1 frame/s for broad discovery, then 5–10 frames/s around selected encounters. Include all relevant small signs when labeling a selected full frame; do not label only the easy large ones. Review all ambiguous labels and a random 10% of straightforward labels in a second pass. For an individual project, record an independent later review rather than pretending there were two annotators.

### 5.6 Split isolation

Freeze train/validation/test before threshold selection. A starting allocation is **70/15/15 by independent groups**, not by frame count.

Group repeated observations of a physical sign site, adjacent frames, repeated trips through that site and closely related route segments. Build the connected grouping constraints before assignment. If repeated commute routes connect almost everything into one group, collect geographically disjoint routes or hold out entire corridors; do not break the grouping constraint just to get the desired percentages.

Reserve additional challenge groups for ramps, LED, zones and glare. They are not a replacement for a representative held-out drive. Training augmentation, synthetic variations and crops inherit their parent split.

A test set reviewed for error-driven model improvement becomes development data for the next version. Obtain a fresh held-out test for the next release claim.

### 5.7 External datasets and privacy

GTSDB provides road-scene detection data; GTSRB provides single-sign classification images. Use each for its actual task, and do not treat either as a Swiss deployment benchmark. Preserve their official benchmark splits where relevant and check the terms for the exact files used. [S13–S14]

Use map and Peugeot TSR outputs only to suggest clips for human review. Agreement between them is not independent ground truth. Do not train a road-applicability model on a map-derived label and then count agreement with that same map as validation.

For sharing, export a separate anonymized demonstration with route IDs removed and faces/plates obscured. Preserve private native-resolution originals locally for evaluation. Do not assume blurring a training copy has no effect on sign-detection results; track it as a dataset transformation.

---

<a id="section-06"></a>

## 6. Training, export and model selection

### 6.1 V100 environment gate

Run all environment, train, evaluate and export commands from the standalone training repository, never from the comma or the openpilot checkout. Keep training separate from the inference runtime on the comma. The supplied source review records a Volta-support difference between PyTorch 2.11 CUDA 12.8/12.9 binaries and CUDA 12.6 builds. Treat that release-specific note as an environment-audit lead, not an evergreen installation command. Check the official packaging information for the exact build being considered; a generic default installation is not an adequate V100 compatibility check. [S15]

Choose a **version-pinned, Volta-compatible build**, starting with an already working supported environment or a CUDA 12.6 build only after its exact package support has been verified. Verify Python and third-party dependencies before locking the full environment. Do not assert that a package works merely because importing `torch` succeeds.

The eventual environment report must include:

```text
GPU name, UUID, device capability and available memory
Driver version and torch.version.cuda
PyTorch build/version and torch.cuda.get_arch_list()
Python version, package lock hash and model-source commit
One actual convolutional forward/backward and optimizer step
Finite loss/gradients; checkpoint save/load; export smoke-test outcome
Peak allocated/reserved memory and competing GPU processes
```

Run FP32 as the numerical reference and evaluate FP16 with gradient scaling. Do not assume native BF16 or TF32 support on V100. A CUDA build’s architecture listing and an actual model step both need to pass. Check the supported architectures of the exact tinygrad backend rather than carrying over PyTorch assumptions; training successfully in PyTorch on a V100 does not establish compatibility with tinygrad `NV`. The runtime reference retained from the source review is a starting point for that check. [S16]

Start on one available GPU. Use a second V100 for independent experiments before adding distributed-training complexity. Do not assume their memory is pooled; distributed training duplicates model state per device unless explicitly implemented otherwise.

### 6.2 Shared preprocessing specification

Define one canonical preprocessing contract, tested independently of the model:

- Decode to an upright native-resolution RGB image with recorded color interpretation.
- For live NV12, respect row strides, plane offsets, color matrix and range. Never treat the camera buffer as tightly packed RGB.
- Preserve the full scene with aspect-ratio-preserving letterboxing. Start with 640 and 960 square detector inputs as experiment candidates.
- Record scale and padding; convert predictions back to original pixels before tracking or cropping.
- Extract classifier crops from the native image, not the downscaled detector image. Start with 96 and 128 pixel classifier inputs.
- Preserve sign borders and associate nearby panels from the original scene.
- Record channel order, tensor layout, dtype, normalization, resizing method and padding value in the model artifact.

Do not reuse the driving model’s warped image tensor merely because it is accessible: its crop/warp may exclude roadside or overhead signs. Likewise, offline HEVC-decoded RGB and live-camera NV12 need a measured domain/parity check, not an assumption of identical pixels.

### 6.3 Bounded experiment matrix

| ID | Detector | Reader | Input | Purpose |
|---|---|---|---|---|
| E0 | None; annotation crops only | Compact classifier | Native crops → 96 | Diagnose whether the value-reading task is learnable. **Oracle crop diagnostic.** |
| E1 | YOLOX-Nano, family classes | Compact classifier | 640; crops 96 | Lowest-cost two-stage baseline. |
| E2 | YOLOX-Tiny, family classes | Same classifier | 960; crops 128 | Measure benefit for small signs against extra runtime cost. |
| E3 | YOLOX-Nano, value-specific classes | Integrated | 640 | One-stage latency/accuracy comparison. |
| E4 | Best compatible candidate | Same selected reader | Fixed chosen shape | Actual target-runtime and thermal comparison. |

Do not launch the full Cartesian product of architectures, sizes and thresholds. Run export/operator smoke tests first; eliminate incompatible candidates before long training.

Use the upstream training recipe as the reproducible starting point, save the complete effective configuration, and first overfit a tiny 32–64-image subset to catch broken labels or preprocessing. Train on real Swiss scenes after that check. Evaluate at least three seeds for the finalist where compute permits, and disclose when only one was run.

Allowed augmentation candidates: moderate brightness/contrast, plausible blur, exposure shifts, scale and perspective variation. Avoid horizontal flips or transformations that corrupt numerals and directional semantics. Synthetic signs are training-only and remain in their parent split.

### 6.4 Selection criteria

Select using validation data and these priorities, in order:

1. Wrong-value and wrong-road accepted-event errors.
2. Useful recall and time available before passage.
3. Performance on small signs, ramps and excluded conditions.
4. Export/runtime correctness and target-device cost.
5. Model size and training convenience.

Model confidence scores are not safety probabilities. Calibrate/reject on held-out validation data; report risk versus coverage. A model that rejects everything must fail the utility gate even though its accepted-event precision is undefined or superficially perfect.

Do not add test-set errors to training while continuing to call the same data an untouched test set.

### 6.5 Export contract

Build a model bundle in local artifact storage, not as committed training weights:

```text
models/detector.onnx     # Required for the selected detector-based pipeline
models/reader.onnx       # Required for two-stage models; absent for a one-stage model
SHA256SUMS              # Hashes of all bundle payloads, not just the weights
manifest.json           # Component roles, versioned I/O and required core contract
labels.json             # Stable class IDs and supported semantic capabilities
preprocessing.json      # Pixel/tensor/coordinate contract; covered by bundle hashes
thresholds.json         # Validation-selected settings; no vehicle-specific magic
training.json           # Training commit/config/dataset/split/seed/environment identities
validation.json         # Selection metrics, coverage and known limitations
model-card.md           # Intended use, unsupported scenarios and evidence status
licenses/               # Code, weight and data provenance obligations
```

Declare the pipeline as `two_stage` or `single_stage` and reject an incomplete component set. The earlier single-file `model.onnx` shorthand must not hide a missing reader model. Checkpoints containing optimizer/RNG state are training-resume assets, not deployment assets.

Export the separately versioned minimal core snapshot from section 4.3 with its source commit/tree hash and compatible contract versions. The runtime's `artifacts.lock.json` pins the bundle digest, core snapshot digest, contract version and locally reviewed capability profile. Compute the bundle digest over the finalized archive; do not put a self-referential archive digest inside the hashed payload. Verify safe extraction, exact allowed members, component hashes and contract compatibility before loading any graph.

For the first target attempt, prefer static input shapes and a portable operator set. Keep NMS and other postprocessing outside the exported graph when that simplifies runtime support. Verify every operator against the **pinned target runtime**, not just the current upstream documentation.

Test PyTorch → ONNX → selected runtime on the same golden images. Compare preprocessed tensors, raw outputs, decoded boxes, class decisions and final event decisions. A successful ONNX checker or model load is not sufficient.

Initial numerical screening thresholds: normalized-output `atol=1e-4, rtol=1e-3` for FP32 and `atol=1e-2, rtol=1e-2` for FP16. These are starting tolerances, not universal guarantees; document any scale-specific adjustment before final evaluation. Above all, adjudicate every change in accepted event, value, applicability or timing. Quantization is excluded until the FP32/FP16 path passes this comparison.

---

<a id="section-07"></a>

## 7. Runtime data contracts

The names in this section are **proposed new interfaces**, not services already present in the fork. Keep the pure Python representation independent of Cap’n Proto so replay tests run without vehicle hardware.

### 7.1 Input and output types

| Type | Required content |
|---|---|
| `FrameRef` | `session_id`, `stream`, `frame_id`, `capture_mono_ns`, capture-reference kind, native dimensions and preprocessing identity. |
| `Detection` | `FrameRef`, original-pixel `bbox_xyxy`, sign family, optional numeric value, optional linked panel boxes, detection/classification scores and supported-domain flags. |
| `ObservationBatch` | `FrameRef`, model/config hashes, processing timestamps, backend status and a bounded list of detections. |
| `RoadContext` | Timestamped calibration/ego-motion availability, road-continuity ID if established, junction/country ambiguity and source provenance. No truth labels. |
| `SignTrack` | Stable track ID, observation IDs, consensus value/family, first/last seen time, applicability evidence and passage evidence. |
| `ValidationResult` | Accepted/rejected flag, stable reason code and validated batch identity; no mutation of the original input. |
| `ApplicabilityEvidence` | `own/other/unknown`, contributing frame/context IDs, method/version and rejection/uncertainty reason. |
| `PassageEvidence` | `ahead/passed/unknown`, optional earliest/latest crossing times in the same session clock, method/version and uncertainty reason. |
| `PerceptionHealth` | Backend availability, last processed capture time, last successful completion time, overflow/fault state and session identity. |
| `SourceSnapshot` | `car/map` provenance, optional existing m/s value, validity, original observation time or explicit age-unknown, and optional road-context identity. |
| `LimitHypothesis` | Optional value, state, source track/event IDs, activation interval, last supporting context time, unavailable/unknown reason and validity flags. |
| `AdvisoryComparison` | Separate snapshots of Car, Map and Vision, source ages, agreement/conflict result and display decision. No target-speed field. |

Numeric values are stored as `value_kph: int | None` in the perception/domain layer. Convert exactly once to `speed_limit_mps: float | None` in the advisory serialization boundary. Display conversion to mph must not alter the original observation or the control offset settings.

### 7.2 `speedVisionState` message proposal

Use a dedicated custom message with three sections: health, recent observations, and the current advisory hypothesis. Start with a maximum of **eight observations/tracks per publication**; report overflow and suppress current-limit decisions for an overloaded scene rather than silently selecting the first sign.

| Section | Fields |
|---|---|
| Identity | `schemaVersion`, `sessionId`, `sequence`, `modelHash`, `configHash`, `rulePackHash` |
| Timing | `publishMonoNs`, `lastProcessedCaptureMonoNs`, `lastObservationMonoNs`, `lastContextMonoNs` |
| Health | `enabled`, `backend`, `backendHealthy`, `inputFresh`, `inferenceMs`, `captureToResultMs`, `droppedFrames`, `overflow`, reason code |
| Observation list | `cameraStream`, `frameId`, `trackId`, normalized box, family, `hasValue`, `valueKph`, scores, applicability, condition status, passage status, capture/last-seen timestamps |
| Hypothesis | `state`, `hasValue`, `speedLimitMps`, `usableForAdvisory`, event/track identity, activation interval, context identity, invalidation reason |

Cap’n Proto fields that cannot be nullable must use explicit presence flags. Zero in a numeric storage field with `hasValue=false` is not a zero-speed limit. Validate this at serialization and consumption.

Do not append Vision to `LongitudinalPlanSP.SpeedLimit.Source` for these increments. Do not reuse existing event identifiers or reserved structs without checking the complete current schema and the fork’s extension convention.

**Publisher ownership:** `speedvisiond` is the sole publisher of `speedVisionState`. With an independent backend, `ObservationBatch` stays inside that process. With Chestnut co-location, introduce a second internal, non-default-logged service, `speedVisionObservations`, carrying the section 7.1 batch contract: the validated GPU-owning runner is its sole publisher and `speedvisiond` its consumer. Do not overload `modelV2` or its driving-model fields with sign detections.

Do not send full camera images through the observation message. If the CPU-side applicability estimator needs the source scene, use a bounded snapshot handoff/cache keyed by the exact `FrameRef`: at most one in-flight snapshot plus one pending snapshot. If the corresponding snapshot is unavailable, mark applicability unknown rather than attach a detection to a newer scene. Account for both snapshots in the memory/latency benchmark. Drop obsolete results rather than grow the cache.

### 7.3 Validity invariants

- A `current` hypothesis requires a supported sign, accepted value, `own` applicability, resolved conditions, verified passage and valid continuity.
- A fresh heartbeat cannot refresh the evidence age of a remembered sign.
- Duplicate frames and two simultaneous camera views of one physical sign cannot be counted as independent temporal observations.
- Out-of-order observations cannot change the active hypothesis retroactively.
- Negative latency, a future capture timestamp, non-finite scores, out-of-bounds boxes or an unknown schema/model contract invalidate the affected result.
- Monotonic timestamps are comparable only inside the same session/clock domain. Keep wall-clock/GPS time separate and map it only through a verified conversion.
- The UI may show a historical observation with an age label, but it cannot continue presenting it as current when `usableForAdvisory=false`.
- No message in this subsystem contains `vTarget`, `aTarget`, button requests, actuation requests or an instruction to change the cruise set speed.

### 7.4 Deterministic API contract

The planned pure interfaces are:

```text
validate_batch(batch, now_mono_ns, expected_session_id) -> ValidationResult
tracker.update(batch) -> tuple[SignTrack, ...]
applicability.evaluate(track, road_context) -> ApplicabilityEvidence
passage.evaluate(track, road_context) -> PassageEvidence
memory.update(tracks, road_context, perception_health, now_mono_ns) -> LimitHypothesis
comparator.compare(car_snapshot, map_snapshot, vision_hypothesis) -> AdvisoryComparison
```

Inject time into these functions; do not call the live system clock in the decision logic. Each result carries a finite, documented reason code. Serialization and live subscriptions are adapters around the pure functions, not hidden dependencies inside them.

---

<a id="section-08"></a>

## 8. Temporal logic, road ownership and passage

### 8.1 Track objects; do not vote across the whole image

Associate detections using camera identity, box geometry, motion continuity, sign family and appearance. Begin with a small, inspectable association algorithm rather than a learned tracker. Test it on overtaking vehicles, repeated signs and neighbouring gantries before adding complexity.

Maintain **multiple candidates plus a separate current hypothesis**. A new candidate must not overwrite the current value merely because its latest confidence is higher. Conversely, a plausible new relevant restriction that cannot be read may make the old current hypothesis uncertain; do not hide that uncertainty by continuing to display the old value as established.

Initial experiment profile:

| Setting | Initial value | Interpretation |
|---|---:|---|
| Detector sampling | 5 Hz; compare 10 Hz | Selected frames, not a requirement to queue every frame. |
| Consensus | 3 consistent observations among the most recent 5 | Same physical track, unique capture timestamps. |
| Evidence window | 0.8 s | Older votes do not accumulate indefinitely. |
| Minimum observation span | 0.2 s | Prevent a burst of duplicate frames from confirming a sign. |
| Track association gap | 0.6 s maximum | Longer gaps require a new track or explicit re-identification. |
| Unconfirmed-track retention | 1.0 s after last observation | Diagnostic retention only; never active evidence. |
| Observation freshness at consumption | 0.35 s maximum | A late result can be logged but must not create a new current event. |

These are starting parameters for validation, not proven operating settings. Evaluate time-to-confirm, not just stability. At 120 km/h, 0.4 seconds spans about 13.3 metres; confirmation plus inference latency can use much of a small sign’s visible approach. This is arithmetic, not a measured recognition distance.

Use validation-set score thresholds and explicitly test confusion pairs such as 30/80, 60/80 and 80/100. Never combine three high-confidence readings of different physical signs into one accepted event. With two cameras, associate shared physical signs before computing temporal evidence; do not double the vote count.

### 8.2 Per-track and current-hypothesis state machines

| Current state | Event | Next state / action |
|---|---|---|
| `unknown` | Supported detection arrives | Create `candidate`; no current value. |
| `candidate` | Temporal value/family consensus succeeds | `confirmed_observation`. |
| `confirmed_observation` | Own-road attribution passes; conditions resolved; sign remains ahead | `confirmed_ahead`; keep current hypothesis separate. |
| `confirmed_ahead` | Passage interval is resolved and continuity remains valid | Create or replace `current` advisory hypothesis. |
| Any candidate state | Attribution is other-road | `rejected`, reason `other_road`; do not lower or raise the current hypothesis. |
| Any candidate state | Family/value/conditions remain unresolved | Keep observation or abstain; never promote it. |
| `current` | Confirmed applicable replacement is passed | Replace with a new event ID; preserve the old event in history. |
| `current` | Applicable cancellation, unresolved relevant restriction, or unresolved branch/country transition | `unknown`, with cause; no guessed replacement value. |
| `current` | Backend/input unavailable or continuity evidence stale | `unavailable` for consumers; retain a clearly historical diagnostic record. |
| Any | New session, restart, reverse-motion ambiguity or invalid clock | Reset current state; do not restore a remembered limit from disk. |

Confirmation of a future higher limit must never create a current higher limit. Use the same passage rule for both increases and decreases; this project is not a braking planner and does not redefine the application point to obtain a more convenient response.

### 8.3 Road and lane applicability: an explicit research task

A sign’s horizontal image position is not enough to decide road ownership. Build an offline baseline using the full frame, available calibration/ego motion, trajectory context, road boundaries and sign/panel geometry. Track the evidence used for each decision.

Implementation sequence:

1. Establish broad ego-road versus clearly-other-road labels and reject ambiguity.
2. Test ramp splits and parallel roads before claiming motorway coverage.
3. Add lane-specific gantry attribution only when lane identity and sign alignment are validated.
4. On turns, merges, splits or lost calibration, reduce certainty or invalidate continuity rather than inheriting the main-road state blindly.

Read-only driving-model lane/path information may be an input, but it is not ground truth and its uncertainty must propagate. Map matching may help identify a road branch, but missing or ambiguous maps cannot be treated as positive evidence of applicability. Never use the map’s speed value to make a camera read more confident.

Until this estimator passes its own holdout test, the product remains an **observation tool**. An oracle road-ownership label can diagnose performance, but cannot satisfy this gate.

### 8.4 Passage estimation: do not equate disappearance with passing

Prototype a geometric passage estimator offline using calibrated temporal observations and ego motion. Candidate evidence includes multi-frame stationary-sign geometry or a reliably observed support/pole location. Quantify uncertainty from calibration, pose and depth. A single assumed sign diameter is not sufficient evidence of exact depth; signs and mounting vary.

Required outputs are `ahead`, `passed` or `unknown`, with an earliest/latest plausible crossing interval where established. A sign leaving the image due to occlusion, cropping, camera pitch or a curve must not imply `passed`.

Validate on labelled approach-and-passage clips, including:

- The sign moves out of view before the vehicle reaches it.
- A truck occludes the sign.
- The vehicle takes the exit rather than the main carriageway.
- A curved road causes the sign to leave the side of the image.
- Two successive signs have the same value.

If passage cannot be established from the available sensors and geometry, display “observed ahead” or “passage unknown.” Do not add a fixed timeout and call it passage detection.

### 8.5 Limit memory and continuity

Keep three different clocks: last processed camera frame, last observed sign, and last evidence of road-context continuity. A remembered sign can outlive its visibility, but must not outlive unresolved road context.

For the initial advisory experiment, use a **maximum unrefreshed hypothesis horizon of 300 seconds or 5 km, whichever occurs first**, in addition to immediate invalidation on unresolved branches, relevant cancellation or lost context. This is a conservative engineering availability cap, not a claim that a legal restriction ends at that time or distance. Record `context_horizon_exceeded` and report its impact on coverage.

Do not integrate distance across missing ego-motion samples. A gap over 0.5 seconds makes distance-based continuity uncertain unless a separately validated adapter resolves it. Treat a route change and a clock reset as different events with distinct reasons.

### 8.6 Swiss rule pack

Create a versioned, reviewed rule pack that maps **observed sign families and context requirements** to interpretation actions. Keep sign recognition separate from jurisdiction rules so future countries do not require relabeling every number.

Each rule record must contain: rule ID, jurisdiction, sign family, required context, action, official source, source revision/date, reviewer and linked tests. The official ASTRA sign catalogue is a suitable visual reference. Legal interpretation must be checked against the effective official text at implementation time. [S17–S18]

For this revision, cancellation, zone, general-limit and conditional-sign interpretation stays disabled until that review is complete. In particular, do not assume that every intersection cancels a restriction or that an end sign means a fixed numerical default. The supplied source review records that the Fedlex landing page was reachable but its operative article text was not reliably extracted. This consolidation does not establish any detailed legal rule; retrieve and verify the effective official text before enabling such interpretation.

---

<a id="section-09"></a>

## 9. Car / Map / Vision comparison and UI

### 9.1 Preserve the operational path

The existing `SpeedLimitResolver`, `SpeedLimitAssist`, operational source enum, set-speed logic and offset calculations remain unchanged for increments A–D.

Build `advisory.py` as a separate read-only comparator. It consumes snapshots of the original outputs and the experimental Vision result. No shared mutable resolver instance, no monkey patch, and no global source-policy change.

**Do not make Vision a third operational SLA source just to show it on the screen**. Adding that source to Assist requires a later specification and regression review.

### 9.2 Explicit comparison table

| Vision state | Car/Map state | Advisory result |
|---|---|---|
| Unusable, absent or stale | Any | Preserve existing SLA display/control behavior; optional separate Vision-unavailable indicator. |
| Readable observation; ownership or passage unknown | Any | “Camera observed X”; no current-limit claim and no overspeed warning. |
| Relevant sign still ahead | Any | Separate ahead indication, not replacement of the current badge. |
| Current, usable hypothesis | Car/Map agree with it | Mark agreement and source identities in diagnostics; show the separate camera hypothesis. |
| Current, usable hypothesis | Car/Map missing | Show a camera-derived current hypothesis with clear provenance; no control action. |
| Current, usable hypothesis | One or both disagree | Show conflict in the diagnostic/advisory view; do not silently take the minimum or always prefer Vision. |
| Applicable cancellation/unsupported relevant context | Car/Map have a value | Vision becomes unknown; the original system continues under its existing policy. |

Agreement is useful diagnostic evidence, not permission to multiply confidence as though the sources were independent. Preserve original source timestamps; re-publication by the comparator cannot make map data younger.

### 9.3 Proposed feature controls

These are new names to check for collisions during integration:

| Setting | Default | Meaning |
|---|---|---|
| `VisionSpeedLimitMode` | `off` | `off`, `shadow`, `observation`, `advisory`; not the legacy `SpeedLimitMode`. |
| `VisionSpeedLimitWarnings` | `false` | Optional advisory warnings only after their separate review. |
| `VisionSpeedLimitDiagnostics` | `false` | Bounded local diagnostic writing. |
| Model capability flags | All unvalidated capabilities false | The bundle identifies whether ownership, passage, rule-pack and backend gates have passed. |

A UI setting cannot override a failed capability/runtime gate. Unsupported settings must result in disabled/observation-only behavior with a reason, not in an exception or an unannounced control fallback.

### 9.4 Display requirements

Use the active existing UI layout. Do not assume one layout covers every supported comma device; the entry point already selects between layouts. [S9]

The first visible release should be a small, separate **Camera** badge and optional ahead marker. Use text/icons as well as styling to distinguish observed, current, conflicted and unavailable states. Keep detailed scores, boxes and rejection reasons in replay/debug views, not permanently over the driving scene.

Do not replace the legacy SLA badge while Assist is using Car/Map, because that could imply it is following the Vision value. The driver must be able to tell which value, if any, the existing system is using.

Warnings remain off by default. They may use only a usable current hypothesis with no unresolved source conflict, and must never be produced from an ahead/observed sign. Add debounce, deduplication and a reviewed warning budget; do not reuse urgent safety alerts or interfere with existing alert priorities. The first advisory release can omit audible warnings entirely.

---

<a id="section-10"></a>

## 10. Device runtime, Chestnut and failure behavior

### 10.1 Two candidate paths, neither assumed feasible

**Path A — Independent local backend:** a separate `speedvisiond` process on a demonstrably available CPU/accelerator. It still competes for CPU, memory bandwidth, power and thermal headroom, so it must pass the full runtime comparison. “Separate process” does not mean “zero impact.”

**Path B — Chestnut-owned inference:** an auxiliary inference module inside the existing validated GPU-owning runner; temporal logic and diagnostics may remain in a separate process. Keep weights, inputs, outputs and configuration independent from the driving model. Do not initialize the same Chestnut GPU from another process. [S2]

The selected runner is part of the support contract. A stock-Python-runner implementation does not imply native `modeld_tinygrad` support. Keep unsupported runner/bundle combinations disabled rather than changing the driving model automatically. [S3]

### 10.2 Scheduling requirements

- Queue at most **one pending frame** and process the latest available frame. Bounded detections and fixed maximum crop work prevent an unexpectedly busy image from expanding runtime without limit.
- Copy or retain camera buffers only with verified lifetime guarantees. A borrowed VisionIPC buffer cannot be used after its producer may reuse it.
- Avoid holding the driving loop while waiting for Vision results, queue space, disk writes or a slow consumer.
- Perform model loading, compilation, memory allocation and warm-up before the validated on-road execution point. No first-use JIT, graph recompilation, model download or unbounded allocation in the driving loop.
- Set a work-admission budget before submitting auxiliary GPU work. Skipping an optional frame is acceptable; queuing it behind a busy driving workload is not.
- Stop admitting new auxiliary work when runtime/thermal conditions exceed the gate. Do not assume already-running GPU kernels can be interrupted safely.
- Never treat a Python thread, process niceness or `try/except` as GPU preemption or fault isolation.

The existing model runner has an `after_enqueue` callback before output synchronization. Its presence is **not evidence that this interval is free for another model**. Any use of that location must account for in-flight driving work, shared queues and synchronization. [S2]

### 10.3 Benchmark instrumentation

Measure separately:

```text
capture -> adapter receipt
receipt -> preprocessing completion
preprocessing -> accelerator submission
accelerator submission -> completion
completion -> postprocessing/consensus
capture -> message publication
publication -> UI consumption
```

Report p50/p95/p99 and maximum observed end-to-end values, along with sample counts and dropped-frame rates. GPU kernel timing alone does not measure USB copies, host queues, synchronization or consumption age.

Also record main-runner timing and deadline misses, camera/frame drops, process restarts, memory, temperatures and available power/thermal telemetry. Fingerprint the driving model, hardware, runtime revisions and experimental bundle in every report.

### 10.4 Proposed runtime gates

First measure the actual main-loop deadline and baseline margin for the selected configuration. Service frequencies are not sufficient proof of a runtime deadline. [S7]

A preliminary budget check is:

```text
main workload + admitted auxiliary workload + required guard margin < deadline
```

Measured percentile sums can screen bad candidates, but are **not a worst-case execution-time proof**. Actual concurrent A/B runs and fault tests are mandatory.

Initial admission targets to freeze before evaluating the candidate:

| Metric | Proposed gate |
|---|---|
| Main-runner new deadline misses attributable to Vision | Zero during the comparison; existing misses must be separately understood. |
| Main-runner p99 regression | No more than the smaller of 1 ms or 2% of baseline p99, assessed across repeated runs. |
| Capture-to-result latency | p95 ≤150 ms and p99 ≤200 ms at the selected sample rate. |
| Freshness | Results over 350 ms old cannot create a new advisory event. |
| Pending frame queue | Maximum one; old frames dropped. |
| Auxiliary output wait in driving loop | None. |
| New allocations/compilations during steady-state processing | None outside approved bounded behavior. |
| Memory | Stable after warm-up; explicit headroom reserved from measured main-workload requirements. |
| Thermal or power throttling caused by Vision | No acceptance until its impact is characterized and the primary runtime gates still pass. |

Run baseline and candidate on equivalent recorded inputs at least three times each, with at least ten minutes of warm-up and thirty minutes of steady-state observation per run. Alternate their order and record ambient/power conditions. Use additional traces for cold start, long drives and unusually busy scenes. These durations are proposed engineering tests, not a safety certification.

If margins do not support the auxiliary model, the correct outcome is observation/replay only or a validated independent backend—not an undocumented reduction in driving-model quality.

### 10.5 Fault response

| Fault | Required Vision behavior | Primary-system boundary |
|---|---|---|
| Model missing/hash mismatch/unsupported ops | Stay disabled; publish diagnostic reason when possible. | Do not select or modify a driving-model bundle. |
| Stale or duplicated camera frames | Stop accepting new sign events; invalidate consumer usability as required. | Do not block camera consumption by the main runner. |
| Detector overload/too many signs | Drop work; set overflow; abstain from current decisions. | No unbounded crop loop or memory growth. |
| Backend timeout or non-finite output | Mark unavailable; latch auxiliary fault for the session. | No attempted recovery that resets a shared GPU during driving. |
| Chestnut transport/GPU failure | Follow the already established primary-system behavior; do not claim isolation. | Test only at bench in this project. |
| Disk full/slow diagnostics | Drop/disable diagnostics, report a bounded counter. | No synchronous disk wait in the driving loop. |
| Missing maps | Preserve Vision’s independent uncertainty; original SLA falls back as before. | No hidden requirement for network access. |
| Restart/model change/session mismatch | Reset Vision memory and require fresh evidence. | No stored limit restored as current. |
| Feature off | Stop auxiliary work and remove its advisory state. | Existing control outputs remain unchanged. |

A timeout cannot guarantee recovery from an in-process driver hang. If this failure mode cannot be bounded without endangering the main runner, Chestnut co-location fails the deployment gate. Make that limitation explicit rather than hiding it behind a watchdog.

### 10.6 Diagnostics and upload isolation

For the initial device experiment, propose `speedVisionState` as a **non-default-logged** service and write optional local sidecar diagnostics. The service registry supports a logging flag, but the complete logger/uploader behavior must still be verified in the fork. [S7]

Select a private path only after proving it is outside automatic route upload, sunnylink upload, backup and file-sharing roots. Do not assume that being outside the route directory alone is enough.

Use bounded rotation, for example ten files of at most 32 MiB each, and stop auxiliary logging when the configured free-space floor is reached. Store observation/event metadata by default, not raw frames. Never leak crops, coordinates or identifiers through normal stdout/cloudlog exception messages. No changes to existing upload preferences are authorized by this project.

---

<a id="section-11"></a>

## 11. Evaluation and release gates

### 11.1 Metrics at the correct level

| Layer | Metrics |
|---|---|
| Detection | Precision/recall by sign family and native-pixel size; false detections in sign-free driving. |
| Crop reading | Value confusion matrix, unreadable rejection and calibration/risk-coverage curves. Oracle crops reported separately. |
| Tracking | Track merges/splits, duplicate accepted events and consensus delay. |
| Applicability | Own/other/ambiguous confusion, wrong-road accepted events and coverage near branches. |
| Passage | Early/late event error relative to annotated crossing intervals; unobservable cases disclosed. |
| Current hypothesis | Accepted-event precision, relevant-event recall, wrong-value duration and distance, unknown coverage and stale-state duration. |
| Runtime | Primary-loop interference, capture-to-result latency, queue bounds, thermal behavior and failures. |

An accepted **current event** is correct only when its value, sign meaning, ego-road relevance, conditions and activation timing are all correct. A correct “80” on the exit ramp is not a correct accepted motorway event.

Count false events on continuous negative driving as events/hour, with distance when available. Count unique physical sites and separate repeated encounters. Frames from the same sign are correlated; they are not thousands of independent successes.

Publish both macro results across scene groups and aggregate results. Always state the supported domain and its coverage. Unknown/unobservable truth is excluded from a specific metric’s denominator only with an explicit count and reason; it cannot disappear from the report.

### 11.2 Proposed advisory acceptance targets

Freeze these before final holdout evaluation. They are engineering gates for an advisory experiment, not evidence of certified road safety.

- Supported-domain accepted-current-event precision: point estimate ≥99%, with a one-sided 95% lower confidence bound ≥99% where independent-event assumptions are defensible.
- Supported-domain relevant-event recall: point estimate ≥90%, with denominator and uncertainty reported. Report the event-level reader-only and full-current-hypothesis recalls separately.
- Wrong-road accepted current events: zero in the predeclared critical challenge suite; disclose all observed errors in ordinary held-out drives as well.
- Premature activation of a higher limit: zero outside the annotated passage uncertainty interval in the acceptance suite.
- End-to-end activation delay: p95 no more than 0.5 s after the upper end of the annotated crossing interval for supported, observable events; report misses separately.
- All deterministic contract, fail-state, mode-isolation and existing SLA regression tests pass.
- All device runtime gates in section 10 pass before on-road shadow use.

For scale: with 300 independent accepted events and zero errors, a one-sided exact 95% lower success bound is approximately 99.01%. Repeated visits to the same sign do not establish that independence. Report route/site-cluster uncertainty as well; insufficient independent data means the statistical gate is not established, not that the denominator can be replaced by frames. Exact binomial intervals are supported by SciPy’s documented methods. [S19]

An observation-only release does not claim to pass current-limit precision or passage gates. It instead reports the performance of what it actually does.

### 11.3 Non-interference has two separate tests

**Semantic:** replay identical recorded perception/planner inputs through the baseline and candidate control paths, with Vision absent, enabled, invalid, conflicted and faulted. Compare actuator requests, operational speed-limit fields, set-speed outputs and control-state transitions. The candidate must not change these semantic outputs. Ignore only documented non-semantic fields such as additional diagnostic timestamps.

**Runtime:** run the actual concurrent workload and compare timing/health. Semantic equivalence does not establish timing safety, and good timing does not prove there is no accidental control dependency.

Test all legacy SLA modes and policy values, metric/imperial display, offsets, PCM/non-PCM configurations relevant to the fork, and transitions while an old limit is remembered. Changing a UI mode must never enable Vision control consumption.

### 11.4 Critical scenario matrix

| Fixture | Expected result |
|---|---|
| Three unique own-road 80 observations; no passage evidence | Confirmed/ahead or observation only; never current. |
| Three observations are actually the same frame | No temporal confirmation. |
| Map says 120; clearly ramp-only sign says 40 | Preserve operational 120 behavior; reject 40 for ego-road advisory. |
| Two lane-specific signs without validated lane association | Unknown applicability; neither becomes current. |
| 80 sign partly occluded so it resembles 30 | No accepted 30 without the required evidence; record the confusion/rejection. |
| 80 sign with unreadable supplementary panel | Observed 80, unresolved conditions, no usable current hypothesis. |
| Cancellation encountered with unresolved default | Vision current becomes unknown; never invent 120. |
| Truck-mounted speed sticker | Reject as an ego-road restriction or mark ambiguous. |
| Healthy heartbeat repeats old detection | Evidence age remains old; no freshness laundering. |
| Frame capture timestamp is in another session | Reject/reset, with clock-domain reason. |
| Camera disappears; old current limit exists | Vision becomes unavailable; original SLA unchanged. |
| New relevant but unreadable restriction ahead | Surface unresolved restriction; do not silently assert the old value is established after its possible application point. |
| Vehicle enters a ramp after main-road recognition | Invalidate/re-evaluate continuity; no blind inheritance. |
| Exported backend changes threshold decisions | Flag every changed event for review; no automatic parity pass. |
| GPU stalls or sidecar queue fills | Optional work stops/drops; no unbounded wait introduced. |
| Feature toggled while legacy Assist is active | No Vision influence on operational target or confirmation behavior. |

---

<a id="section-12"></a>

## 12. Repository file maps, scripts and test boundaries

All paths and commands below are implementation targets. They are not existing files or tested commands. Use the two explicit repository roots; never interpret an unqualified training path as a path in openpilot.

### 12.1 Repository roots and common conventions

| Symbol used in this plan | Meaning |
|---|---|
| `TRAIN_REPO` | Local root of the proposed `cristianku/openpilot-nnslr-tools` repository. |
| `RUNTIME_REPO` | Isolated local checkout/worktree of `cristianku/openpilot`; not the checkout used for driving. |
| `NNSLR_DATA_ROOT` | Private local dataset/run/evaluation directory outside both repositories. |

All core, data, training and offline replay tests in T1–T6 run from `TRAIN_REPO`. T7 has a pure comparator test in `TRAIN_REPO` and control-isolation tests in `RUNTIME_REPO`; T9–T12 integration tests run in `RUNTIME_REPO`. Test commands are labeled accordingly. T0 must correct any runtime path-layout mismatch in the map and commands together.

### 12.2 Required standalone training-repository layout

```text
openpilot-nnslr-tools/
  README.md
  AGENTS.md
  CONTRIBUTING.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  .gitignore
  .dockerignore
  .env.example
  requirements/
    cpu.lock
    train-v100.lock
  docker/
    Dockerfile.tools
    Dockerfile.train-v100
    compose.yaml
  docs/
    plan.md
    setup.md
    data-collection.md
    annotation-guide.md
    training.md
    evaluation.md
    export-and-integration.md
    troubleshooting.md
    vision-speed-limit/
      audit.md
      decisions.md
      rulepack-ch.md
    model-cards/
      template.md
  configs/
    datasets/ch_pilot.yaml
    models/yolox_nano_reader.yaml
    train/v100_baseline.yaml
    evaluation/observation.yaml
    evaluation/current_hypothesis.yaml
    local/                         # Ignored; private paths and host settings
  schemas/
    route-manifest.schema.json
    frame-manifest.schema.json
    annotation.schema.json
    dataset.schema.json
    bundle-manifest.schema.json
  src/
    nnslr_tools/
      __init__.py
      cli.py
      environment.py
      sync.py
      manifest.py
      extract.py
      annotations.py
      dataset.py
      models/
      train.py
      evaluate.py
      export.py
      replay.py
      hard_examples.py
      bundle.py
      adapters/openpilot_routes.py # Optional, pinned, local route decoder
    speed_vision_core/
      __init__.py
      types.py
      preprocess.py
      tracking.py
      applicability.py
      passage.py
      state.py
      advisory.py
  scripts/
    check_environment.py
    sync_routes.py
    extract_frames.py
    import_annotations.py
    validate_dataset.py
    build_splits.py
    train.py
    evaluate.py
    mine_hard_examples.py
    export_onnx.py
    replay.py
    package_model.py
    verify_bundle.py
    export_core.py
  tests/
    core/
    data/
    training/
    cli/
    fixtures/                      # Synthetic or explicitly redistributable only
  .github/workflows/
    ci.yaml
    gpu-validation.yaml           # Manual, trusted execution only
```

`docs/plan.md` must contain this complete specification, with its current revision. `README.md` is the executable quickstart and navigation entry point, not another divergent plan. Script wrappers dispatch into tested `nnslr_tools` functions; do not duplicate training or preprocessing implementations in the wrappers.

### 12.3 Documentation and environment requirements

| File | Required content and verification |
|---|---|
| `README.md` | Purpose, independent-repository boundary, explicit experimental/advisory status, CPU synthetic quickstart, authorized V100 workflow, script index, data placement and artifact handoff. |
| `AGENTS.md` | Module ownership, test commands, no control-path changes, no auto-download/train/push/deploy, no private data in Git, source/config/data provenance, and requirement to update tests/docs with behavior changes. |
| `docs/setup.md` | Isolated native and container workflows; tested versions and lock hashes; one selected GPU; memory/disk checks; competing-job checks; no host-driver or power-limit modification. |
| `docs/data-collection.md` and `annotation-guide.md` | Selected video plus matching log requirements, timing validation, box/value/panel/road/passage labels, negative scenes, privacy, manual review and no annotation truth at inference. |
| `docs/training.md` | Tiny-overfit diagnostic, scratch versus pretrained initialization, detector versus reader stages, effective configuration, random seeds, class imbalance, rejected augmentations, checkpoints, exact resume behavior and OOM recovery. |
| `docs/evaluation.md` | Frozen site/route splits, validation-only model selection, observation versus current-event metrics, negative driving, challenge sets, confidence bounds, and test-set contamination handling. |
| `docs/export-and-integration.md` | Multi-component bundle schema, reference-core snapshot, hashes, export parity, target compatibility, explicit runtime promotion and no automatic vehicle deployment. |
| `docs/troubleshooting.md` | Missing video/logs, bad alignment, incorrect labels, unsupported GPU build, nonfinite gradients, corrupted resume checkpoint, export drift and incompatible contract. Each has a diagnostic and a non-destructive recovery path. |
| `CONTRIBUTING.md`, notices and model-card template | Fixture/privacy requirements, local tests and review process, separately checked code/model/data licenses, known limitations and evidence-status fields. Repository visibility/license are owner decisions. |

Lock the CPU and V100 dependency profiles separately after testing them. Container bases must be pinned by immutable digest after the compatibility audit. A CPU environment must be usable without installing CUDA or importing PyTorch in the pure core/data tests; GPU dependencies belong only to the explicit training profile. A Python lock alone is not a CUDA-compatibility proof.

Container data mounts are read-only for originals and writable only for derived/run outputs. Use the caller's UID/GID, an explicit selected GPU, and no privileged mode or host Docker socket. Do not pass through all GPUs by default. Default examples must not overwrite existing directories or checkpoints.

Ignore private local config, `.env`, dataset paths, videos, logs, derived images, run folders and large checkpoints in Git and container build contexts. Enforce an allowlist for redistributable tiny fixtures so the smoke tests still have meaningful sample inputs.

### 12.4 Required script contracts

Each script exposes `--help`, validates inputs before expensive work, logs the effective configuration and identities, returns nonzero on failure, and refuses destructive overwrite by default. Prefer the same options in `nnslr` CLI subcommands and Python wrappers. Remote access and GPU work are explicit commands, never hidden import-time or setup-time side effects.

| Script | Required inputs and behavior | Output / failure requirements |
|---|---|---|
| `check_environment.py` | Default CPU/read-only package and space checks; explicit `--gpu-smoke --device 0` performs the separately authorized actual forward/backward/export test. | Machine-readable environment report; distinguish unavailable, untested and failed GPU support. |
| `sync_routes.py` | Explicit host, route/segment allowlist, camera streams, local destination and `--dry-run`; only selected video and matching logs. | Inventory, transferred-file hashes and resume information. Never `rsync --delete`, change remote software, download the whole device by default or embed SSH keys. |
| `extract_frames.py` | Local route manifest or ordinary image input; explicit camera/timing profile; bounded sampling. | Frames plus timestamp/provenance manifest. Missing or ambiguous clock joins cannot silently become valid temporal samples. |
| `import_annotations.py` | Declared annotation-tool format and matching frame identities. | Canonical objects/events with importer version, unmapped-field report and unresolved-label flags. Never infer truth from map/current-speed values. |
| `validate_dataset.py` | Canonical manifests/annotations and supported vocabulary. | Bounds/labels/panel/timing checks and counts; invalid training inputs fail, while explicitly unknown truth remains representable. |
| `build_splits.py` | Validated dataset, site/encounter/route grouping and seed. | Immutable split definitions, counts/leakage report and portable dataset manifest. An impossible leakage-free requested ratio is reported, not forced. |
| `train.py` | Full model/train config, portable dataset manifest, explicit device/seed/run directory, and optional `--resume`. | Resolved config, source/environment/data identities, train/validation metrics, selected checkpoint and resumable checkpoint. Validation—not test—selects weights. |
| `evaluate.py` | Pinned checkpoint or exported bundle, frozen split and named capability profile. | Metrics with denominators, exclusions, uncertainty, false events and challenge-set outcomes. Oracle-crop reports are explicitly labeled. |
| `mine_hard_examples.py` | New unlabeled or train/validation pool plus candidate predictions and uncertainty/error criteria. | Deduplicated manual-review queue; never silently promotes predictions to truth or mines the frozen test set for retraining. |
| `export_onnx.py` | Selected checkpoint/config and fixed I/O contract. | Every graph required by the pipeline, preprocessing/labels and parity report. A two-stage detector without its trained reader is rejected. |
| `replay.py` | Local frames/video/timing/context plus verified bundle and named capability profile. | Annotated replay and observation/event JSONL; no truth-injected runtime inputs. |
| `package_model.py` | Export outputs, evaluation reports, model card, license records and core-contract identity. | Local immutable candidate bundle plus digest. Refuse missing reports, component mismatches or private-path leakage. Does not publish or install. |
| `verify_bundle.py` | Candidate bundle/digest and optional target contract. | Safe-archive/hash/schema/component checks and explicit compatibility results; validation does not initialize the main driving system. |
| `export_core.py` | Clean selected source commit and a manifest of allowed pure-core files/tests/licenses. | Minimal core snapshot and hashes; reject a snapshot containing train/data/remote-access code or unrecorded local changes. |

Training must record the resolved initialization source and its checksum, loss/optimizer/scheduler configuration, detector and reader stages, batch/accumulation settings, precision, dataloader configuration and random states. A resume checkpoint includes model, optimizer, scheduler and enabled scaler/RNG states plus dataset/split/config identities. Reject a mismatch rather than silently starting a different experiment under the same run ID. Record any permitted resume differences and the reproducibility limits; do not promise bit-identical training across different hardware or dependency builds.

**Proposed CLI workflow, to implement and verify in T1–T5; not runnable merely because it appears here:**

```bash
# From TRAIN_REPO, after setup.md has installed the tested isolated environment.
# A committed synthetic fixture makes the first data/CLI smoke test independent
# of private routes. Actual V100 commands run only with approved compute access.
python scripts/check_environment.py --report "$NNSLR_DATA_ROOT/runs/environment.json"
python scripts/validate_dataset.py --dataset "$NNSLR_DATA_ROOT/manifests/dataset.v001.json"
python scripts/train.py \
  --config configs/train/v100_baseline.yaml \
  --dataset "$NNSLR_DATA_ROOT/manifests/dataset.v001.json" \
  --device 0 --seed 17 \
  --run-dir "$NNSLR_DATA_ROOT/runs/e1-seed17"
python scripts/evaluate.py \
  --run-dir "$NNSLR_DATA_ROOT/runs/e1-seed17" \
  --split validation --profile configs/evaluation/observation.yaml
python scripts/export_onnx.py \
  --run-dir "$NNSLR_DATA_ROOT/runs/e1-seed17" \
  --output "$NNSLR_DATA_ROOT/runs/e1-seed17/export"
```

The scripts must first reject an unset/empty `NNSLR_DATA_ROOT` and unsafe output locations; the documented quickstart includes explicit data-root setup, fixture preparation and directory creation before these commands. `train.py` performs the environment gate itself or verifies a matching valid report; a CPU-only environment report never authorizes or proves a GPU workload.

A model's frozen release evaluation is a separate step after selection, as specified in T8. The `validation` command above is not a release test and does not unlock vehicle use.

### 12.5 Runtime-only file map

Paths in this table are relative to `RUNTIME_REPO`, using the historical package layout until T0 verifies it.

| Path | Responsibility |
|---|---|
| `openpilot/sunnypilot/speed_vision/__init__.py` | New integration package; no framework initialization at import. |
| `openpilot/sunnypilot/speed_vision/_vendor/speed_vision_core/` | Approved, hash-pinned pure-core snapshot generated from TRAIN_REPO; no handwritten divergence. |
| `openpilot/sunnypilot/speed_vision/adapters.py` | Live frame/read-only-context adapters for the pure core. |
| `openpilot/sunnypilot/speed_vision/runner.py` | Selected-backend interface, coordinate/parity handling and bounded scheduling. |
| `openpilot/sunnypilot/speed_vision/speedvisiond.py` | Optional lifecycle, publishing and explicitly private diagnostics. |
| `openpilot/sunnypilot/speed_vision/artifacts.lock.json` | Approved bundle/core identities, schema version and local capability profile; no credentials or automatic update URL. |
| `openpilot/sunnypilot/speed_vision/tests/` | Core-snapshot parity, messaging, control isolation, faults, lifecycle and UI/integration tests. |
| `openpilot/cereal/custom.capnp`, `log.capnp`, `services.py` | Existing files; advisory-message registration only after schema review. |
| `openpilot/system/manager/process_config.py` | Existing optional-process lifecycle after runtime selection. |
| `openpilot/selfdrive/modeld/modeld.py` | Existing file; only if the actual selected stock Chestnut path passes review. |
| Actual parameter registry/UI widget paths found in audit | Existing files; modify only identified active paths. |

**No training directory is added to the runtime repository.** Dataset curation, PyTorch architecture/training, ONNX export and general replay tools belong in TRAIN_REPO. Target-only benchmark/control-regression harnesses may remain in RUNTIME_REPO because they test that specific integration.

Protected operational files are read and regression-tested, not changed: `speed_limit_resolver.py`, `speed_limit_assist.py`, relevant planner/control consumers and PSA/opendbc safety/control code.

### 12.6 Test ownership and CI

```bash
# TRAIN_REPO: no private data, GPU or built vehicle stack needed.
python -m pytest tests/core tests/data tests/cli -q

# TRAIN_REPO: CPU training/export tests use the separately installed test profile;
# actual GPU tests remain explicitly selected after compute authorization.
python -m pytest tests/training -m "not gpu" -q

# RUNTIME_REPO: only after the audited runtime test environment is available.
python -m pytest openpilot/sunnypilot/speed_vision/tests/test_control_isolation.py -q
```

Add import-isolation, CLI-help/config-validation, schema, leakage, missing-component, checksum-mismatch and clean-setup tests. Shared-core golden fixtures must produce equivalent reference and vendored-core outputs at the pinned revision. Test model artifacts before any approved runtime import.

`ci.yaml` runs the core/data/CLI suite with synthetic fixtures and checks for private-file leakage. `gpu-validation.yaml` is manual and restricted to trusted reviewed revisions and explicitly authorized compute; never execute untrusted pull-request code with home-server credentials or private dataset mounts. No workflow deploys to the comma or publishes route data. Record whether a report is CPU-only, a mock backend, an actual V100 test or an actual target-device test.

---

<a id="section-13"></a>

## 13. Dependency graph and approval gates

```text
T0 source/environment audit
   ├─> T1 standalone repo + contracts -> T2 alignment -> T3 dataset/splits
   │                                               |
   │                                      T4 train/export -> T5 perception replay
   │                                                               |
   └─> approved backend feasibility work                 T6 temporal interpretation
                                                                   |
                                                        T7 advisory comparison
                                                                   |
                                                        T8 frozen evaluation
                                                                   |
                                 backend approved ------> T9 device integration
                                                                   |
                                                        T10 bench non-interference
                                                                   |
                                                        T11 supervised shadow
                                                                   |
                                                        T12 advisory UI/release
```

No GPU feasibility workload starts merely because the source audit is approved. Obtain permission for the specific host/device and workload; use existing logs/read-only status first.

| Gate | Evidence required | Unlocks |
|---|---|---|
| G0 — Audit accepted | Immutable reference revisions, explicit two-repository ownership/name decision, local workspace, selected-runner status or unknown, data inventory and proposed first increment. | Local offline implementation in the training repository; remote creation/publication remains separately authorized. |
| G1 — Data trustworthy | Alignment report, annotation validation, leakage tests and frozen split. | Meaningful model comparison. |
| G2 — Perception useful | Model bundle, export parity, negative-driving results and held-out observation metrics. | Observation-only replay demonstration. |
| G3 — Interpretation established | Non-oracle ownership, passage, rule-pack and continuity tests with appropriate coverage. | Current-hypothesis evaluation/advisory capability. |
| G4 — Runtime feasible | Explicitly authorized benchmark; target operator support, measured headroom and backend decision. | Bench integration only. |
| G5 — Non-interference accepted | Semantic replay, full legacy-mode matrix, concurrent runtime tests, faults and verified disable/rollback procedure. | Separately authorized shadow collection. |
| G6 — Shadow accepted | Representative shadow reports, failure review, privacy verification and no unexplained primary-system regressions. | Observation UI; current advisory only if G3 also passed. |
| G7 — Advisory release accepted | Current-hypothesis acceptance report where enabled, UI tests, support matrix and reviewed known limitations. | The scoped advisory release, not vehicle control. |

A gate is marked **not established** when data is insufficient. Do not turn missing measurements into a pass or silently lower a threshold after seeing the test results.

<a id="section-14"></a>

## 14. Task-by-task execution backlog

All test/module names below are implementation targets. They do not exist merely because they appear in this document. Repository roots and canonical module ownership are defined in section 12; do not recreate training tools in the runtime fork. For each approved code task: first add the specified test/fixture, observe its failure, implement the minimum behavior, run the targeted and neighbouring regression tests, and make a small local commit. Do not push or deploy automatically.

### T0 — Read-only audit and first-increment decision

**Dependencies:** None; source inspection in the approved planning workspace.

**Interface:** This specification and the accessible checkout → evidence-backed `audit.md`, `decisions.md`, identified tests and a proposed offline increment.

**Local output:** `docs/vision-speed-limit/audit.md` and `decisions.md` in the approved planning workspace, destined for TRAIN_REPO. If that repository has not yet been created, keep the documents locally without creating a remote. Remote targets remain read-only.

- [ ] Confirm the standalone-training-repository requirement and proposed `openpilot-nnslr-tools` name, owner and visibility. Record whether it exists and which local workspace is approved; do not create or publish it during T0.
- [ ] Read repository/parent instructions; record branch, commit, uncommitted changes and submodule status without changing them. Read the NNLC reference README for workflow structure only; do not assume its training stack or targets suit vision.
- [ ] Re-check the source-snapshot observations and exact files listed in section 2, plus their relevant schema/parameter/UI/upload consumers. Record discrepancies rather than assuming the snapshot is still accurate.
- [ ] Record existing host/device access and unknowns. Do not connect to a device or inspect a GPU host unless that access is explicitly authorized in the execution session.
- [ ] Propose the first ten local approach-and-passage clips and the tests for T1–T3; stop for review before implementation or workload execution.

Useful local read-only commands, from the RUNTIME_REPO root:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
git submodule status
git grep -n 'SpeedLimitResolver\|speedLimit.resolver\|SpeedLimitMode' -- openpilot
git grep -n 'modeld_tinygrad\|is_stock_model\|is_tinygrad_model' -- openpilot/system/manager
git grep -n 'narrowRoadEncodeIdx\|wideRoadEncodeIdx' -- openpilot
```

**Acceptance:** every fact has a source/command result; actual device state is never inferred from the GitHub branch. Record the real existing SLA regression-test command for reuse in every integration task.

### T1 — Standalone repository, domain contracts and synthetic fixtures

**Dependencies:** T0/G0 accepted.

**Interface:** Approved repository/workspace decision plus section 7 contracts → an installable local training-tools package, CPU quickstart, hardware-independent reference core and synthetic fixtures used by T2–T7.

**Repository:** TRAIN_REPO.

**Files:** Root README/AGENTS/packaging/ignore/CPU-environment files, `docs/plan.md`, `src/nnslr_tools/cli.py`, `src/speed_vision_core/types.py`, `tests/core/test_contracts.py`, `tests/cli/test_entrypoints.py`, `tests/cli/test_import_isolation.py` and `tests/fixtures/`. Section 12 defines the remaining files to add with their owning tasks; do not ship empty implementations disguised as working commands.

- [ ] Initialize the separately approved local repository/workspace, add packaging, docs and private-file exclusions, and register the CLI with a real synthetic quickstart. Remote repository creation and a push happen only when separately authorized.
- [ ] Write failing tests for core-only import without the training framework/openpilot, CLI help and malformed inputs; implement the smallest useful package, then re-run from a clean isolated environment.

- [ ] Encode the types and invariants in section 7, including explicit absence and clock domains.
- [ ] Add fixtures for valid observations, unknown values, non-finite scores, bad boxes, session mismatch, duplicates and stale evidence.
- [ ] Keep data objects importable without importing CUDA, tinygrad or messaging.
- [ ] Verify serialization round-trips preserve absence/unknown and never reinterpret zero as an accepted limit.

```bash
# From TRAIN_REPO
python -m pytest tests/core/test_contracts.py tests/cli/test_entrypoints.py tests/cli/test_import_isolation.py -q
```

**Acceptance:** a clean local clone has the instructions and a working CPU/synthetic quickstart without a comma or full openpilot checkout. All malformed cases have deterministic rejection reasons; raw km/h and serialized m/s are not mixed. No training is implemented inside the runtime tree; unfinished later scripts are documented as not implemented, not reported as tested.

### T2 — Local manifest and frame alignment

**Dependencies:** T1; selected local video/log access authorized.

**Interface:** Selected route files → `routes.jsonl`, `frames.jsonl`, extracted frames and an alignment report under the section 5 storage contract.

**Repository:** TRAIN_REPO.

**Files:** `src/nnslr_tools/manifest.py`, `extract.py`, optional `sync.py` and route adapter; script wrappers `sync_routes.py`, `extract_frames.py`; `tests/data/test_manifest.py`, `test_alignment.py`, and explicit-transfer tests for any implemented sync command.

- [ ] Start with fixture files containing a missing segment, a duplicate frame ID and differing video/log frame counts.
- [ ] Implement local-only inventory and verified joins using the pinned route/encoding interfaces.
- [ ] Reject remote paths, traversal and unresolvable alignment; support classification-only extraction with an explicit non-timing status.
- [ ] Produce the ten-clip visual alignment report before processing a large dataset.

```bash
# From TRAIN_REPO
python -m pytest tests/data/test_manifest.py tests/data/test_alignment.py -q
```

**Acceptance:** rerunning extraction preserves identity; gaps and uncertain timing remain visible in the manifest.

### T3 — Annotation validator and split builder

**Dependencies:** T2; annotations available for the selected frames/encounters.

**Interface:** Frame manifests and reviewed object/event annotations → validated dataset views and a frozen, hashed split with leakage/coverage reports.

**Repository:** TRAIN_REPO.

**Files:** `src/nnslr_tools/annotations.py`, `dataset.py`; wrappers `import_annotations.py`, `validate_dataset.py`, `build_splits.py`; schemas and annotation docs; `tests/data/test_annotations.py`, `test_splits.py`. Dataset manifests/annotations remain outside Git.

- [ ] Enforce pixel-coordinate bounds, explicit unreadable values and supplementary-panel associations.
- [ ] Build encounter/site/route grouping constraints and detect cross-split leakage.
- [ ] Test a connected-commute case that cannot meet a 70/15/15 split without leakage; report the need for more data rather than breaking the group.
- [ ] Freeze and hash the pilot split; report encounter counts, unique sites, negative-driving duration and coverage gaps.

```bash
# From TRAIN_REPO
python -m pytest tests/data/test_annotations.py tests/data/test_splits.py -q
```

**Acceptance:** no related frames, physical sites or derived crops cross the declared split boundary.

### T4 — Training/environment and export baseline

**Dependencies:** T3/G1; specific GPU workload authorized. Target-device probing also needs its own authorization.

**Interface:** Frozen training/validation data and preprocessing contract → trained reference checkpoint, section 6.5 model bundle and export/parity evidence.

**Repository:** TRAIN_REPO.

**Files:** `src/nnslr_tools/environment.py`, `models/`, `train.py`, `export.py`, `bundle.py`; `src/speed_vision_core/preprocess.py`; V100 configs/lock/container profile; check/train/export/package/verify script wrappers; setup/training/export docs; `tests/core/test_preprocess.py`, `tests/training/test_resume.py`, `test_export_parity.py`, `test_bundle.py`. Implement all relevant section 12 script contracts, including finite-step checks, explicit initialization and resumable state.

- [ ] After compute authorization, run the actual V100 environment smoke test and save its evidence.
- [ ] Verify RGB/NV12/letterbox/crop-coordinate contracts with known synthetic images before training.
- [ ] Run the tiny-subset overfit check; then compare E1–E3 on the frozen split using recorded configurations.
- [ ] Produce the full model bundle and pre-/postprocessing parity report; reject unsupported target operators before expensive refinement.

```bash
# From TRAIN_REPO, using the approved training/export test environment
python -m pytest tests/core/test_preprocess.py -q
python -m pytest tests/training/test_resume.py tests/training/test_export_parity.py tests/training/test_bundle.py -q
```

**Acceptance:** a checkpoint that merely loads is insufficient. Include actual finite training, export execution, golden-output comparisons and validation-based selection.

### T5 — Perception replay and event evaluation

**Dependencies:** T4; implementation must not use annotation truth as runtime input.

**Interface:** Frames, model bundle and read-only recorded state → observation JSONL, annotated replay and perception/event metrics; truth enters evaluation only.

**Repository:** TRAIN_REPO.

**Files:** `src/nnslr_tools/replay.py`, `evaluate.py`, `hard_examples.py`; corresponding script wrappers and evaluation docs; `tests/data/test_replay.py`, `test_metrics.py`, `test_hard_examples.py`.

- [ ] Separate oracle crops/truth-assisted diagnostics from the normal runtime-input path.
- [ ] Export annotated video, timestamped observation JSONL and a negative-driving report.
- [ ] Test event matching so repeated frames do not multiply successes and a wrong-road value cannot count as a correct current event.
- [ ] Reproduce the same results with the same input/model/config hashes and report any nondeterministic backend variation.
- [ ] Implement the hard-example review queue on new/train/validation pools, with a regression test proving frozen-test identities cannot enter the retraining queue. Predictions remain unreviewed suggestions until a human accepts the labels.

```bash
# From TRAIN_REPO
python -m pytest tests/data/test_replay.py tests/data/test_metrics.py tests/data/test_hard_examples.py -q
```

**Acceptance:** increment A is demonstrable without any vehicle integration or current-limit claim.

### T6 — Tracking, applicability, passage and memory

**Dependencies:** T5 for end-to-end evaluation; pure fixture work can follow T1 after approval.

**Interface:** `ObservationBatch`, `RoadContext` and `PerceptionHealth` → `SignTrack`, applicability/passage evidence and `LimitHypothesis` using section 7.4.

**Repository:** TRAIN_REPO.

**Files:** Canonical `src/speed_vision_core/tracking.py`, `applicability.py`, `passage.py`, `state.py`; corresponding tests in `tests/core/` and rule records in `docs/vision-speed-limit/rulepack-ch.md`.

- [ ] Implement multi-track consensus with injected time, deduplication and bounded history.
- [ ] Add explicit other-road/ambiguous outcomes; evaluate the non-oracle estimator on ramp and parallel-road holdouts.
- [ ] Validate passage against annotated intervals; retain `unknown` when geometry cannot establish it.
- [ ] Implement the transition table, context-loss rules and engineering availability horizon; link every rule to a fixture.

```bash
# From TRAIN_REPO
python -m pytest tests/core/test_tracking.py tests/core/test_applicability.py tests/core/test_passage.py tests/core/test_state.py -q
```

**Acceptance:** replay explains every promotion, replacement, rejection and invalidation. Human truth is not used as a runtime input. Failed G3 leaves current-hypothesis capability disabled.

### T7 — Advisory comparison and protected-path tests

**Dependencies:** T6; audit-recorded legacy regression commands available.

**Interface:** Car/Map `SourceSnapshot` values and Vision `LimitHypothesis` → `AdvisoryComparison`; original operational outputs remain unchanged.

**Repositories:** TRAIN_REPO for `src/speed_vision_core/advisory.py` and `tests/core/test_advisory.py`; RUNTIME_REPO for the audited read-only regression harness in `openpilot/sunnypilot/speed_vision/tests/test_control_isolation.py`. This task may use a local candidate core snapshot on recorded/synthetic input; it does not register a live service or install anything on the comma.

- [ ] Implement the exact comparison table in section 9 without modifying the operational resolver.
- [ ] Exercise every legacy mode/policy with Vision absent, ahead, current, invalid, stale and conflicting.
- [ ] Compare baseline and candidate operational resolver/control outputs, including remembered limits, units and offsets.
- [ ] Check static imports/subscriptions and dynamic replay for unintended Vision consumption by Assist/planners.

```bash
# From TRAIN_REPO
python -m pytest tests/core/test_advisory.py -q

# Separately, from RUNTIME_REPO in its audited test environment
python -m pytest openpilot/sunnypilot/speed_vision/tests/test_control_isolation.py -q
```

**Acceptance:** complete the existing SLA suite recorded in T0 as well; no changed operational values or confirmation behavior.

### T8 — Freeze thresholds and issue the offline report

**Dependencies:** T5–T7 complete for the capability being evaluated.

**Interface:** Frozen model/config/domain and untouched test data → observation/current-hypothesis reports and explicit per-capability pass/fail/not-established outcomes.

**Repository:** TRAIN_REPO for evaluation profiles, approved sanitized summaries/model cards and pipeline scripts. Full `evaluation/v001/` reports remain in NNSLR_DATA_ROOT outside Git. Record training/core/model identities and the compatibility contract, not only a model filename.

- [ ] Freeze the selected bundle, thresholds, supported domain and acceptance definitions.
- [ ] Run the untouched test once for the release decision; report denominators and site/route correlations.
- [ ] Review every accepted wrong event and all early-higher-limit cases; distinguish defects from insufficient data.
- [ ] Publish pass/not-established/fail for each capability. Do not retune on the same holdout and present it as a fresh evaluation.

**Acceptance:** increment B produces a reproducible report, not just a curated successful video.

### T9 — Device backend and messaging, bench only

**Dependencies:** T8 for the scoped offline capability; G4 plus explicit bench-integration approval.

**Interface:** Validated live frame/context adapters and selected backend → bounded inference, sole-owner observation publishing and optional `speedVisionState` lifecycle.

**Repositories:** RUNTIME_REPO for `runner.py`, `adapters.py`, `speedvisiond.py`, the generated `_vendor/speed_vision_core/`, `artifacts.lock.json`, approved existing schema/manager files, and `test_runner.py`, `test_message_contract.py`, `test_lifecycle.py`, `test_core_parity.py` in the integration tests. TRAIN_REPO owns `export_core.py` and its source/hash/dependency-isolation tests.

- [ ] Export and verify the selected core snapshot and complete model bundle; pin both identities and the contract in the runtime lock. Prove the runtime can import the core without installing nnslr_tools, PyTorch training dependencies or a training checkout.
- [ ] Run golden-fixture reference/vendored-core comparisons and component/contract mismatch tests before initializing the live backend.

- [ ] Select Path A or B using authorized measurements and the actual runner/bundle fingerprint.
- [ ] Implement one-frame queueing, fixed maximum work, optional lifecycle and dedicated advisory messages.
- [ ] Disable all unsupported hardware/runner/model combinations. Audit local-only diagnostic storage and logging.
- [ ] Add fake-backend tests for delay, overflow, non-finite output, unavailable hardware and shutdown; then run the separately marked real-backend bench tests.

```bash
# From RUNTIME_REPO
python -m pytest openpilot/sunnypilot/speed_vision/tests/test_runner.py openpilot/sunnypilot/speed_vision/tests/test_message_contract.py openpilot/sunnypilot/speed_vision/tests/test_lifecycle.py openpilot/sunnypilot/speed_vision/tests/test_core_parity.py -q
```

**Acceptance:** no added required service in the main control-health checks; feature-off does not initialize the auxiliary model or consume its GPU resources.

### T10 — Concurrency, faults and rollback proof

**Dependencies:** T9; authorized bench and shared-resource fault tests.

**Interface:** Baseline/candidate artifacts on equivalent input → semantic, timing, fault and rollback reports required for G5.

**Files:** bench harness and `tests/test_faults.py`, `test_runtime_gates.py`; private benchmark reports.

- [ ] Execute the repeated A/B benchmark with the same selected main model and representative input sequences.
- [ ] Inject stale frames, blocked consumers, slow/full logging, restarts and backend failures at the bench.
- [ ] Test the off switch, invalid-bundle rejection and restoration of the previous deployment artifacts while the vehicle is parked or on a non-driving bench.
- [ ] Re-run semantic isolation and existing SLA regressions after the final integration diff.

```bash
python -m pytest openpilot/sunnypilot/speed_vision/tests/test_faults.py openpilot/sunnypilot/speed_vision/tests/test_runtime_gates.py -q
```

**Acceptance:** reports contain actual timings and fault behavior. A passing mock suite alone cannot pass G5.

### T11 — Explicitly approved shadow collection

**Dependencies:** G5 accepted and the specific shadow collection authorized.

**Interface:** Bench-tested artifact in shadow mode → private field summaries and G6 review; no Vision UI, warnings or control effects.

**Output:** private session summaries and per-scene failure review.

- [ ] Confirm the bench-tested artifact, source/model hashes and feature mode before each session.
- [ ] Collect passively with no Vision badge, warnings or control effects; no debugging interaction while driving.
- [ ] Review primary-loop health, latency, rejected/false events, privacy and road coverage after each session.
- [ ] Stop further shadow testing on unexplained primary-system regression; reproduce at the bench before resuming.

**Acceptance:** G6 review records the actual tested domain and remaining gaps. Do not claim all-weather or all-runner support from one motorway session.

### T12 — Advisory UI and release bundle

**Dependencies:** G6 for observation UI; G3 and final G7 review additionally required for current-hypothesis capability.

**Interface:** `AdvisoryComparison` plus validated capability/health flags → separate Camera display, release documentation and a tested disable/rollback procedure.

**Files:** active UI/state/parameter files identified in T0; `tests/test_advisory_ui.py`; release notes/model card.

- [ ] Add a separate Camera badge with observed/ahead/current/unavailable states and explicit provenance.
- [ ] Verify both relevant layout paths, source-conflict display, stale-state removal and feature-mode transitions.
- [ ] Keep current-hypothesis display gated by G3 and G7; keep audible warnings disabled unless separately reviewed.
- [ ] Package the support matrix, training/runtime/core/model/data provenance, known limitations, disable/rollback procedure and all gate reports. Keep TRAIN_REPO docs/scripts/model cards canonical for reproduction; RUNTIME_REPO release notes identify the precise consumed artifacts. No training checkpoint or dataset enters the device release.
- [ ] Prepare an optional upstream/community proposal with the architecture, measured results, limitations, licenses and a separately anonymized demonstration. Publication or a pull request requires explicit approval; private routes and raw data stay local.

```bash
python -m pytest openpilot/sunnypilot/speed_vision/tests/test_advisory_ui.py openpilot/sunnypilot/speed_vision/tests/test_control_isolation.py -q
```

**Acceptance:** increment D remains advisory-only under every legacy SLA mode. A future operational Vision source requires a new specification.

<a id="section-15"></a>

## 15. Effort, decision records and risk ownership

Do not commit to a delivery date before T0 establishes data availability and runtime constraints. Track four budgets independently: engineering time, annotation/review time, training/benchmark GPU time and safe opportunity to collect representative routes.

For a first scheduling discussion, allow approximately **1–2 engineering days for the audit, 3–5 for data tools/contracts, 3–5 for a perception baseline/replay, and 5–10 for the initial interpretation/evaluation work**, excluding annotation and data collection. These are planning allowances, not promises. Device integration can add substantial work if the runtime path is unsupported; estimate it only after the backend feasibility result. Observation-only delivery remains available if road attribution/passage take longer.

The first hundred labelled frames should establish actual annotation throughput before extrapolating to the full pilot. Benchmark a short training run before estimating GPU hours. Do not assume adding a second V100 halves elapsed project time.

| Risk | Early decision/test | Safe project outcome if unresolved |
|---|---|---|
| Wrong-road signs dominate motorway errors | Ramp/parallel-road holdout and event review in T6. | Observation-only; no current hypothesis in that scene class. |
| Passage is not observable reliably | Annotated uncertainty intervals and non-oracle test. | Ahead/observed display only. |
| Chestnut has insufficient margin or shared-fault risk | Early authorized operator/timing probe, then G4/G5. | Offline tool or measured independent backend. |
| Active runner differs from assumed runner | T0 manager/bundle trace. | Unsupported combination disabled; no forced model change. |
| Dataset is one repeated commute | Site/route split graph in T3. | Collect disjoint routes; report limited generalization. |
| Runtime/compiler alters small-sign decisions | Golden preprocessing/export/event parity. | Keep reference backend; change candidate before deployment. |
| Unsupported conditions are confidently misread | Mandatory panels/unknown classes and negative scenes. | Abstention and explicit unsupported capability. |
| Advisory data enters control through shared state | T7 static/dynamic isolation, repeated in T10/T12. | Reject integration until outputs are separated. |
| Private evidence reaches upload/file-sharing paths | T0/T9 route/log/share-root audit. | Disable auxiliary diagnostics until isolation is established. |
| Training/runtime divergence | Core snapshot hashes, artifact contract, shared golden fixtures and two-root tests. | Reject a mismatched bundle; keep the previously approved runtime or leave Vision disabled. |
| Repository is documentation-only or scripts rely on unrecorded setup | T1 clean CPU/synthetic quickstart and T4 approved real training/export exercise. | Do not mark the training-toolkit milestone complete until the scripts and instructions are exercised. |

Record consequential choices in `decisions.md`: the alternatives, evidence, chosen option, supported domain and revisit condition. Runtime, road attribution and passage decisions must never be justified only by a successful demo.

<a id="section-16"></a>

## 16. Codex handoff prompt

Give Codex this file and the following prompt. No previous conversation or earlier plan is needed. The first task is an audit, not deployment.

```text
Use NN_VISION_SPEED_LIMIT_ALL_IN_ONE_v4.md as the complete specification for
the NN Vision Speed Limit project. Its design, tasks, gates, source references
and execution boundaries are embedded in that file; no earlier plan is needed.

For THIS FIRST TASK, perform T0 only. Repository/source inspection is read-only;
write only the local audit/decision documents in an approved planning workspace.
Do not begin implementation, training, GPU benchmarks or deployment yet.

REPOSITORY BOUNDARY IS MANDATORY:
- Proposed new training repository: cristianku/openpilot-nnslr-tools
  (NNSLR = Neural Network Speed Limit Recognition). The name/visibility are
  proposals pending owner confirmation; do not create or publish it in T0.
- Runtime integration repository: cristianku/openpilot
- Organizational reference only: cristianku/openpilot-nnlc-tools

All setup/annotation/training/evaluation/export instructions, executable scripts,
configs, tests and the pure speed_vision_core reference belong in the standalone
training repository. A folder inside openpilot is NOT an acceptable substitute.
Training ordinary image datasets must not require a full openpilot checkout.
The runtime consumes pinned model/core artifacts and owns only device-specific
adapters, scheduling, messaging, controls-isolation tests and advisory UI.
Do not import the training checkout, install its training stack on the comma,
or copy NNLC lateral-control targets into the vision model.

Runtime reference branch: psa-torque-sunny-testing
Historical source-snapshot commit: 4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9
Verify the checkout/revision instead of assuming it matches that reference.
Read all applicable AGENTS.md and repository instructions first.

Do not checkout, pull, reset, update submodules, install packages, change GPU
power limits, restart services or modify the comma. Device/GPU-host access
requires explicit authorization in this execution session; otherwise keep
those facts unknown and continue with the code audit.

Verify the recorded source observations in section 2 against the actual checkout.
They were inherited from an earlier source review, not live device inspection.
Pay particular attention to:
1. Which runner is selected: stock Python modeld versus native modeld_tinygrad.
2. Chestnut ownership, frame lifetime, enqueue/synchronization behavior and
   whether any auxiliary work could block the driving model.
3. Actual narrow/wide camera/encode-index names and their clock semantics.
4. Existing Car/Map resolver policies and all downstream Assist consumers.
5. Schema extension conventions, optional-process lifecycle, active UI layouts,
   parameter registry, uploader, backup and file-sharing roots.
6. Existing tests and the exact command for the current SLA regression suite.

Preserve the architecture boundary: Vision is a separate advisory result.
Do NOT add Vision to the operational SpeedLimitResolver or source enum, inject
it into carStateSP.speedLimit, change offsets/confirmation logic, or modify
PSA CAN/radar/longitudinal/safety code. Do not alter driving-model weights.

Produce docs/vision-speed-limit/audit.md in the approved local planning
workspace, destined for TRAIN_REPO, with:
- owner/name/visibility decision and existing-versus-proposed repository status;
- the two-root ownership map, standalone CPU quickstart and script deliverables;
- explicit model/core artifact-contract and promotion boundaries;
- immutable source and submodule identities;
- facts supported by file/line or command evidence;
- actual hardware/access status, clearly separated from unknowns;
- selected-runner and candidate-backend findings;
- exact frame/log alignment sources and identified clock domains;
- actual schema, parameter, lifecycle, UI and upload integration paths;
- available local data, gaps and first ten proposed test clips;
- existing regression commands and proposed T1–T3 test fixtures;
- capability gates that are blocked and how to establish them;
- a small first offline implementation proposal and its acceptance criteria.

Present the audit and stop for review. Do not claim training or runtime tests
were run unless they were actually authorized and executed. Do not fabricate
hardware readings, metrics, filenames, branch contents or successful tests.

After an increment is approved, implement T1 in the separate approved local
training workspace; preserve this complete specification as docs/plan.md.
Do not create or push a remote repository without explicit authorization.
Use an isolated runtime worktree/branch for integration and test-first changes. Execute one independently reviewable task at a time, preserving all
constraints in the plan. Never use annotation truth as deployed perception
input. No push, vehicle installation or control integration is implicit in an
approval for offline development.
```

<a id="section-17"></a>

## 17. Sources and evidence provenance

References S1–S19 are carried forward from the supplied plans. Those plans record runtime-repository inspection at the pinned commit and documentation review on 17 September 2026; this revision has not independently repeated those inspections. Reference S20 was read through the GitHub connector for this repository-organization update. Keep the links for traceability, and record fresh file/line or command evidence in T0 before relying on repository-specific details. Re-check version-sensitive framework documentation when selecting an environment.

Proposed APIs, thresholds, architecture, schedules and tests are design decisions, not measured results or claims copied from these sources. Detailed Swiss legal interpretation remains unverified and disabled pending the rule-pack review. References are embedded here so an executor does not need the earlier documents.

**[S1] Branch reference recorded by the earlier source review**
```text
https://api.github.com/repos/cristianku/openpilot/git/ref/heads/psa-torque-sunny-testing
```

**[S2] GPU owner and stock model-runner implementation**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/selfdrive/modeld/modeld.py
```

**[S3] Process manager and selected-runner routing**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/system/manager/process_config.py
```

**[S4] Operational Car/Map speed-limit resolver**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py
```

**[S5] Assist state and target-speed logic**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py
```

**[S6] Existing custom schemas and operational source enum**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/cereal/custom.capnp
```

**[S7] Stream names, service frequencies and logging/decimation configuration**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/cereal/services.py
```

**[S8] Route file definitions and local/remote route handling**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/tools/lib/route.py
```

**[S9] Python UI entry point and layout selection**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/openpilot/selfdrive/ui/ui.py
```

**[S10] Submodule layout**
```text
https://github.com/cristianku/openpilot/blob/4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9/.gitmodules
```

**[S11] sunnypilot’s public SLA documentation**
```text
https://community.sunnypilot.ai/t/speed-limit-assist/249
```

**[S12] YOLOX authors’ repository: baselines and deployment tooling**
```text
https://github.com/Megvii-BaseDetection/YOLOX
```

**[S13] Official GTSDB dataset description**
```text
https://benchmark.ini.rub.de/gtsdb_dataset.html
```

**[S14] Official GTSRB dataset description**
```text
https://benchmark.ini.rub.de/gtsrb_dataset.html
```

**[S15] PyTorch release/packaging information on Volta support**
```text
https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290
https://github.com/pytorch/pytorch/releases
https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337
```

**[S16] tinygrad runtime documentation; not a substitute for testing the pinned fork**
```text
https://docs.tinygrad.org/runtime/
```

**[S17] ASTRA official Swiss sign catalogue**
```text
https://www.astra.admin.ch/fr/signaux
```

**[S18] Official Swiss signalling ordinance landing page** — reference for the future legal-rule review; operative article text not verified in this revision.
```text
https://www.fedlex.admin.ch/eli/cc/1979/1961_1961_1961/en
```

**[S19] SciPy documentation for exact binomial confidence intervals**
```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
```

**[S20] User's NNLC tools repository: organizational reference, read for revision 4**
```text
https://github.com/cristianku/openpilot-nnlc-tools/blob/main/README.md
```
The README was retrieved through the connected GitHub file-read action on 17 September 2026. It describes a standalone tool repository with setup instructions, Python data utilities and Julia NNLC training. This supports the naming/workflow comparison only; no NNLC scripts were executed or copied. `main` is a moving reference, so an executor must record an immutable commit before reusing any implementation. NNSLR repository layout, scripts and core/artifact ownership are new design proposals, not facts from this README.

---

**First concrete deliverable:** ten correctly synchronized Swiss clips, a leakage-checked pilot dataset, and a reproducible replay that reads signs and explains its abstentions. The next question is not whether it can draw a box around “80”, but whether it can reject an exit-ramp “40”, preserve timestamps and uncertainty, and run without changing the primary system.


---

<a id="appendix-a"></a>

## Appendix A. Configuration and threshold register

This register gathers the settings defined in the specification. It is **not** a ready-to-load configuration file, an installation command, or proof that a parameter is supported. Every deployed profile must be hashed, capability-gated and validated. Interpret these values together with their full section; do not copy an isolated threshold into the operational control stack.

| Item | Proposed initial value or gate | Controlling section |
|---|---|---|
| Feature mode | `off`; shadow/observation/advisory require their gates | 9.3, 13 |
| Warnings / private diagnostics | Both off by default | 9.3–9.4, 10.6 |
| Pilot annotation budget | 2,000–5,000 frames; 300–500 encounters; report unique sites separately | 5.5 |
| Continuous negative footage | At least 60 minutes for the pilot, not a deployment-reliability claim | 5.5, 11.1 |
| Data allocation | Start at 70/15/15 by independent connected groups, not frames | 5.6 |
| Detector input candidates | 640 or 960, aspect-ratio-preserving letterbox | 6.2–6.3 |
| Reader input candidates | 96 or 128; crop from native-resolution image | 6.2–6.3 |
| Tiny-subset training check | 32–64 images before the main training comparison | 6.3 |
| Finalist repeatability | At least three seeds where compute permits; disclose fewer | 6.3 |
| Detector rate | Compare 5 Hz with 10 Hz; no backlog | 8.1 |
| Temporal consensus | 3 of the last 5 consistent, unique-timestamp observations | 8.1 |
| Evidence window / minimum span | 0.8 s / 0.2 s | 8.1 |
| Association gap / candidate retention | 0.6 s / 1.0 s after last observation | 8.1 |
| Observation freshness | At most 0.35 s for a newly accepted advisory event | 8.1, 10.4 |
| Hypothesis availability cap | 300 s or 5 km without renewed qualifying evidence, whichever occurs first; not a legal expiry rule | 8.5 |
| Ego-motion gap | More than 0.5 s makes distance continuity uncertain unless independently resolved | 8.5 |
| Maximum published observations/tracks | Eight; flag overflow and suppress current decisions on overload | 7.2 |
| Pending frame queue | One, latest-frame-only | 10.2 |
| Scene snapshot handoff | One in-flight plus one pending, keyed to exact `FrameRef` | 7.2 |
| Capture-to-result latency | p95 ≤150 ms; p99 ≤200 ms | 10.4 |
| Main-runner p99 regression | At most the smaller of 1 ms or 2% of baseline p99 | 10.4 |
| Additional attributable primary deadline misses | Zero during the declared comparison | 10.4 |
| Repeated runtime comparison | At least 3 baseline and 3 candidate runs; 10 min warm-up + 30 min steady state each | 10.4 |
| Diagnostic rotation example | 10 files × 32 MiB maximum, plus a measured/configured free-space floor | 10.6 |
| Accepted-current-event precision | ≥99% point estimate and ≥99% one-sided 95% lower bound where independence assumptions are defensible | 11.2 |
| Relevant-event recall | ≥90% point estimate, with uncertainty and exclusions reported | 11.2 |
| Wrong-road / early higher-limit challenge failures | Zero in the predeclared relevant challenge suite; disclose ordinary-drive errors too | 11.2 |
| Current-event activation delay | p95 ≤0.5 s after the annotated crossing interval’s upper bound, for supported observable events | 11.2 |

**Values intentionally chosen from evidence rather than invented here:** class-confidence thresholds, calibrated road-ownership thresholds, exact GPU memory reserve, main-loop deadline, target driver/package versions, device diagnostic path/free-space floor and any audible-warning budget. The responsible task must measure/select and record these before its gate can pass; their absence is not a default permission to deploy.

<a id="appendix-b"></a>

## Appendix B. Deliverables, approval record and release checklist

### B.1 Required outputs by increment

| Increment | Minimum deliverable bundle |
|---|---|
| A — Offline perception | Separate training repository with documented and exercised CPU quickstart plus all implemented stage scripts; source/environment audit; frame/alignment manifests; validated annotations/split; preprocessing/model identities; observation JSONL; annotated replay; held-out observation and negative-driving reports. |
| B — Offline interpretation | Tracking/ownership/passage/continuity tests; reviewed rules for every enabled semantic capability; event timeline; Car/Map/Vision comparison; non-oracle metrics; explicit capability decisions. |
| C — Device shadow | Approved backend/runner support matrix; bounded message/lifecycle implementation; semantic and concurrent-runtime comparisons; fault/rollback proof; privacy audit; passive session summaries. |
| D — Advisory UI | Separate Camera UI; observed/ahead/current/unavailable/conflict tests; current capability gates where enabled; feature defaults; release/model documentation; known limitations; tested disable/rollback procedure. |

These are outputs to produce during execution, not additional prerequisite documents. Keep public summaries separate from private videos, locations and detailed route evidence.

### B.2 Approval record

For each task or gate, append a record to the project’s local audit/decision documents with the following fields. Unestablished facts remain explicitly unestablished rather than being filled with guesses.

| Field | Required record |
|---|---|
| Scope | Task/gate ID, increment and requested capability. |
| State | `not_started`, `in_progress`, `blocked`, `ready_for_review`, `accepted` or `rejected`. Planning alone never means accepted. |
| Authorized work | Repository/workspace, host/device, permitted operations and explicit approval evidence. |
| Artifact identities | Code/submodule commits, model/config/rule-pack hashes and dataset/split hashes relevant to the work. |
| Evidence | Exact commands, exit status, report paths, observations and test coverage; mark mock-only evidence. |
| Decision | Reviewer, date, accepted scope or rejection reason, unresolved conditions and the next permitted action. |

Initial execution state for **T0–T12 and G0–G7: not started / not established**. The supplied plans’ historical code-review notes are audit inputs, not completed project gates.

### B.3 Release and rollback checklist

- [ ] Record the training-repository commit, pure-core snapshot hash, model-bundle/component hashes, contract version and runtime-repository commit. Verify no training dependencies or private data are included in the device package.
- [ ] Re-run the clean training-repository quickstart and verify README/script/config consistency before publishing any toolkit release; remote publication requires explicit approval.
- [ ] Record supported device, backend, selected runner, driving-model bundle and source/submodule revisions; disable unmatched combinations.
- [ ] Record supported sign families, scene domain, unresolved conditions and whether ownership/passage/current-hypothesis capabilities passed.
- [ ] Attach model/config/rule-pack provenance, license obligations and frozen evaluation reports with denominators and uncertainty.
- [ ] Include commands and evidence for targeted tests, existing SLA regressions, semantic isolation and actual concurrent runtime tests.
- [ ] Verify feature-off behavior, missing/invalid bundle behavior, private logging limits and no automatic evidence upload.
- [ ] Preserve the previous deployment artifacts and identify their hashes before any separately authorized device change. Do not use a destructive Git reset as a generic rollback mechanism.
- [ ] Document and bench-test disabling the auxiliary feature, restoring the prior approved artifact, and checking the main process/control behavior afterward. All artifact changes occur while parked or on the bench, never during driving.
- [ ] Keep restoration from private data backup separate from software rollback; neither operation should delete raw routes or overwrite uncommitted work.
- [ ] Review the Camera badge and conflicts so the displayed advisory estimate cannot be mistaken for the source that legacy Assist actually follows.
- [ ] Keep audible warnings off unless separately reviewed. Vision never becomes an operational cruise target under this specification.
- [ ] Publish only an explicitly approved, anonymized demonstration or upstream proposal; publication is not automatic release work.

**Start here:** hand this file to Codex and run the section 16 audit-only prompt. The first implementation decision follows the T0 evidence review; it does not require another planning document.
