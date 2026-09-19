<!-- [reviewed-dataset] - START -->
# Canonical reviewed dataset v1

The executable contract is `nnslr_tools.annotations`; the portable structural
schema is [annotation.schema.json](../schemas/annotation.schema.json).
The CLI uses stdlib only. Dataset truth stays outside the runtime contract.

## Records and provenance

Each JSONL record is one reviewed object, or an explicitly confirmed negative
frame (`sign_family=not_a_sign`, `bbox_xyxy=null`, `value_state=not_applicable`).
Coordinates refer to the original PNG, not the review preview. Fields are
required, unknown fields rejected, and absent evidence is represented by null
or a declared unknown enum, never zero speed or an invented timestamp.

`annotation_id` is a stable hash of route, segment, camera, output index and
object identity; the source decoded index takes precedence when known.
`output_index` and `decoded_frame_index` are different fields.
Legacy image-only records may have no decoded index; they retain the actual
image SHA-256 and remain usable only for tasks not requiring capture timing.
A decoded index requires a source-video hash. Exact timing additionally needs
a positive capture timestamp, source-log hash and explicit SOF/EOF provenance.

Families: maximum_speed, cancellation, zone, variable_display, other_sign,
not_a_sign, unreadable, road_marking. The last two preserve unreadable-family
and reviewed asphalt evidence; a road marking is not silently promoted to a
vertical regulatory sign. Values are read/unreadable/unknown/not_applicable;
a read value is an integer in the existing core domain 5..250 km/h. This numeric
domain is a plausibility constraint, not a claim about local road law.
Applicability is own_road/other_road/unresolved. Panel presence, visibility,
occlusion, temporary status and variable-display status remain independent.

Review states are pending/accepted/corrected/rejected. Pending data is invalid
for this dataset. Accepted/corrected records are eligible reviewed labels;
rejected proposals remain audit records and never become positive labels or
negatives automatically. Only explicit whole-frame confirmation with zero
retained detections creates a negative frame. Deleted proposals are retained as
rejected records alongside it. The importer compares the reviewed proposal with
its original to distinguish acceptance from correction. Exact original JSONL
inputs are archived by SHA-256 under `annotations/imports/`.

Every imported object preserves reviewer, timezone-aware review timestamp,
notes, proposal source/score, original proposal/model and source-review hash.
These fields record the review assertion; the tool does not authenticate the
reviewer's identity. A browser export is not dataset truth until import passes.
Legacy browser confirmations lacking reviewer/time must be reconfirmed.

## Validation

`validate-dataset` checks schema/types/enums, finite values, numeric domain,
pixel bounds, image existence/hash/geometry, complete PNG chunks/checksums/scanlines,
duplicate identities, contradictory whole-frame negative/object labels, review state,
provenance and consistent monotonic capture evidence. It reports errors as JSON
and exits nonzero. No corrupted input is silently coerced into a valid label.
Missing capture timing remains unresolved and is counted separately from exact
timing; label validation does not close the T2 synchronization gate.

## Frozen splits

The split builder unions full routes with shared site IDs, encounter IDs and
image hashes before assignment. Empty/unknown grouping labels do not join
unrelated routes. Supply the same `site_group` in the review UI for repeated
visits to the same physical site, and `encounter_id` for shared encounters.
Missing physical-site labels are reported: route isolation alone cannot prove
that two trips never pass the same sign.

At least five connected groups are needed for all five requested partitions.
One complete group containing reviewed negatives is reserved for hard-negative
assessment, including its positive controls; another is route-held-out. The
remaining groups use approximately 70/15/15 train/validation/test, with at least
one group in each. Allocation is by group, not image count. The seed defaults
to zero, and sorting plus hashing makes repeated invocations reproducible.
Rejected audit records are excluded. Requested isolation is never weakened to
force a ratio when grouping constraints leave too few independent samples.

Split filenames are hashes of their canonical contents; changing a dataset
requires a new split. Existing named split files cannot be overwritten with a
different definition. Validation with `--splits` detects stale dataset hashes,
missing/extra members, unknown split names and group leakage.

## Current gate

Synthetic import/validation/split tests demonstrate tooling, not a trained Swiss
model or a human-reviewed Swiss corpus. Human annotation, independent visual
synchronization measurements, dataset coverage and a frozen model I/O contract
are still required before the baseline training milestone.
<!-- [reviewed-dataset] - END -->

<!-- [model-review] - START -->
## Automatic review and candidate data

`--dataset-kind training-candidate` explicitly permits model-reviewed rows in
`import-annotations`, `validate-dataset` and `build-splits`. Its default output
is `<data-root>/datasets/training_candidate_dataset/objects.jsonl`; gold remains
the default mode and accepts human review only. Gold validation rejects model
rows even when imported directly as canonical annotations. Candidate split
metadata cannot be presented as gold, and all five route/site/image-disjoint
partitions are still required.

Human canonical rows retain schema version 1. Model rows use version 2 and
require `provenance.label_basis=model_review`, a source-review SHA256,
`review_model`, immutable `review_model_version`, `review_model_sha256`, finite
`confidence` in [0,1], a matching `reviewer=codex/<review_model>`, an aware review
timestamp and verified image hash. Reviewed-frame imports carry model fields
at the top level; the importer transfers them to canonical provenance.
Pending/unresolved reviews cannot be imported. Model confidence is a model
assertion, not calibrated accuracy or human ground truth.

Run the automatic reviewer **on the data host**, with an already-running local
llama.cpp vision server and verified model metadata:

```sh
python -m nnslr_tools.model_review /path/to/preannotations.jsonl \
  --data-root /path/to/data \
  --output /path/to/data/review/run-name \
  --model-metadata /path/to/review-model.json \
  --server-metadata /path/to/review-server.json
```

`--limit 3` produces an explicitly sampled pilot. Without a limit all supplied
frames are reviewed. The endpoint is loopback-only; no image goes to a remote
API. Pillow comes from the existing optional `preannotate` dependency profile.
The worker does not download models, launch servers, access devices or start
training. Server metadata binds PID/argv/port to the actual process and the
model/projector files; model metadata includes model ID, pinned revision,
asset filenames/SHA256s, directory and bundle SHA256 (SHA256 of the sorted,
compact JSON asset-hash dictionary).

The current input is the existing bootstrap preannotator, **not a trained
NNSLR predictor**. Every audit row marks this comparison scope and records
`nnslr.available=false`. Thus the resulting discrepancy counts are not NNSLR
precision/recall measurements. Ordinary `other_sign` proposals are outside
the speed comparison vocabulary but remain in the original evidence.

Pass 1 sees the full frame without prediction labels. Pass 2 sees a full frame
plus an unlabeled proposal crop, or four overlapping tiles covering the frame.
This is a different visual method using the same reviewer, so its errors are
correlated. Both raw responses, transforms, prompts, times and model identity
are retained. Multiple objects, conflicting confident answers and unreadable
values remain unresolved. Only confirmed negatives and single objects whose
proposal crop is explicitly associated with the answer become candidates.
False negatives with no verified box stay audit-only. No coordinate is invented.

Each run keeps `run.json`, immutable input/model/prompt/source binding,
per-frame checkpoints, and `final/report/{audit.jsonl,report.html,run.json}`
with category manifests. `final/candidate_reviews.jsonl`, a separate canonical
candidate dataset, `validation.json` and `splits.json` record downstream
results. Rerunning the same command resumes completed frames; a changed
binding requires a new output directory. Missing inputs or failed model calls
stop the run with an explicit UNAVAILABLE checkpoint, which is retried on
resume. Failed/insufficient splits are reported without randomizing adjacent
frames. No model review closes missing capture-time provenance, the human
gold-set gate, Swiss coverage or the unimplemented training/evaluation stages.
<!-- [model-review] - END -->
