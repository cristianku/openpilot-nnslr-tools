VIDEO PROCESSING REQUIREMENT — IMPLEMENT, DO NOT ONLY DESIGN

T2 must include executable tooling for real comma camera recordings.

Supported inputs must include:

- fcamera.hevc
- ecamera.hevc
- qcamera.ts when available
- qlog
- rlog

Use ffmpeg/ffprobe as the primary video decoding/probing backend unless there
is a demonstrated reason to use something else.

Implement real tools/modules for:

1. VIDEO INVENTORY

Inspect a downloaded comma route/segment and report:

- available camera files
- codec
- resolution
- nominal FPS
- duration
- frame count when reliably available
- timestamps/time base
- corruption/truncation warnings
- stream identity

CLI:

nnslr video-probe <segment-or-route>

2. FRAME EXTRACTION

Implement extraction from comma HEVC video.

Support:

- full sequential extraction
- extraction at fixed sampling rate, e.g. 1 fps
- extraction over a specific time range
- extraction around candidate events
- preserving original image resolution
- deterministic frame naming
- no silent frame resizing

CLI examples:

nnslr extract-frames \
  --video fcamera.hevc \
  --output <dir> \
  --fps 1

nnslr extract-frames \
  --video fcamera.hevc \
  --start 120.0 \
  --end 135.0 \
  --fps 10

3. FRAME MANIFEST

Every extracted frame must have a machine-readable record containing at least:

route_id
segment_id
camera_stream
source_video
decoded_frame_index
presentation_timestamp
capture timestamp when established
encode/frame id when established
native_width
native_height
image_path
alignment_status
alignment_error_ns or equivalent uncertainty field

Do not infer capture time from FPS if the log provides better evidence.

4. COMMA LOG ALIGNMENT

Implement readers/adapters for the relevant qlog/rlog metadata needed to align
video with:

narrowRoadEncodeIdx
wideRoadEncodeIdx
narrowRoadCameraState
wideRoadCameraState

Establish and document the exact relationship between:

video frame
presentation order
encode index
frameId
timestampSof
timestampEof

Do not pretend this relationship is proven until verified on real route data.

CLI:

nnslr align-route \
  --video <fcamera.hevc> \
  --log <qlog-or-rlog> \
  --stream narrow_road \
  --output alignment.jsonl

5. ALIGNMENT REPORT

Produce a report including:

- matched frames
- unmatched video frames
- unmatched log frames
- duplicate ids
- timestamp discontinuities
- segment boundary problems
- decoder errors
- partial GOP handling
- estimated timing error

CLI:

nnslr alignment-report <alignment.jsonl>

6. CANDIDATE CLIP EXTRACTION

Implement tooling that can use existing log metadata only as a SEARCH HINT,
never ground truth.

For example, transitions in:

liveMapDataSP.speedLimit
liveMapDataSP.speedLimitAhead

may suggest candidate timestamps.

Then create clips around those times, e.g.:

-30 seconds ... +30 seconds

CLI:

nnslr find-candidates <route>
nnslr make-clips <candidate-manifest>

Candidate source must be recorded explicitly, e.g.:

candidate_reason = "liveMapDataSP_transition"

The map speed must NEVER become the camera-label ground truth.

7. DATASET FRAME SELECTION

Support:

broad discovery:
~1 fps

around selected sign encounters:
5–10 fps

Do not blindly extract/store every 20 fps frame for training.

Frames belonging to one physical encounter must carry the same encounter_id so
later split logic can prevent train/test leakage.

8. OUTPUT LAYOUT

Use NNSLR_DATA_ROOT, never a hardcoded machine path.

Suggested layout:

$NNSLR_DATA_ROOT/
  raw/
    routes/
  derived/
    frames/
    clips/
    alignment/
  manifests/
  annotations/
  runs/
  evaluation/

Raw downloaded video/log files must never be committed to Git.

9. TESTS

Create synthetic/small generated video fixtures where possible and tests for:

- frame extraction
- start/end range
- deterministic naming
- missing video
- corrupt video
- missing log
- frame/log count mismatch
- duplicate encode index
- timestamp discontinuity
- segment transition
- partial GOP
- alignment uncertainty reporting

Tests must not require comma hardware.

10. DOCUMENTATION

README/docs must include an end-to-end example:

comma route files
    ↓
video-probe
    ↓
align-route
    ↓
extract-frames
    ↓
find-candidates
    ↓
make-clips
    ↓
human annotation
    ↓
build-dataset
    ↓
split-dataset
    ↓
training on V100

The goal is that once I copy/download new comma videos into NNSLR_DATA_ROOT,
the repository provides the scripts/CLI needed to transform them into a
training-ready dataset without manually writing ad-hoc ffmpeg commands.