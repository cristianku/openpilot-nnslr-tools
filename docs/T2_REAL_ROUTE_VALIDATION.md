<!-- [t2-validation] - START -->
# T2 — Local real-route validation, 2026-09-19

Status: **partial; the T2/G1 gate is not closed**. This report covers local
TRAIN_REPO processing only. No runtime modification, container deployment,
GPU workload or device installation was performed for this increment.

## Input and reproducibility

Four consecutive local segments, each with `fcamera.hevc` and full `rlog.zst`.
Private recordings, decoded images and generated clips remain outside Git.
The narrow camera decodes at its native 1344 × 760 resolution. The HEVC
bitstream is approximately 20 Hz, but ffprobe reports an average rate of 25;
using that average as a capture clock would be incorrect.

Configure `NNSLR_OPENPILOT_ROOT`, `NNSLR_OPENPILOT_PYTHON` and, when needed,
`NNSLR_OPENDBC_ROOT` as described in the README. The selected parser interpreter
needs the optional `requirements/log-reader.txt` profile; no runtime imports
are required in the core/CLI or parser child.

The real local run exercised `route-manifest`, `video-probe`, `log-metadata`,
`align-route`, `find-candidates`, `make-clips`, and frame extraction. Route-wide
1 Hz extraction was already exercised on the container before this increment:
198 PNGs (59 + 60 + 60 + 19). Those existing manifests remain media-only.

## Observed results

| Segment | Full-rate decoded frames | Aligned frames | Unresolved | Map candidates projected |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1180 | 0 | 1180 | 0 |
| 1 | 1200 | 1200 | 0 | 2 |
| 2 | 1200 | 1200 | 0 | 3 |
| 3 | 385 | 385 | 0 | 0 |
| Total | 3965 | 2785 | 1180 | 5 |

“Aligned” means a unique zero-based `EncodeIndex.segmentId` join with a
positive SOF/EOF capture timestamp. It does **not** claim an independent
measurement of camera latency or a completed visual synchronization audit.

All raw HEVC PTS/best-effort timestamps were absent. Per-frame duration was
0.05 seconds; the separately identified `raw_hevc_frame_duration` media clock
allows candidate projection without inventing capture timestamps.

Five H.264 review clips were generated and decoded with ffprobe. All retain
1344 × 760 geometry and contain 655, 1035, 1137, 1137 and 983 frames respectively.
Requested windows use the existing 30-second before/after defaults and are
naturally limited by each source segment. Two candidates refer to simultaneous
changes of different map fields, so their clips are duplicates of the same
interval: five candidate records represent four distinct review centers.

The four distinct center PNGs were visually inspected: valid native road
images, with urban crossings and a road-marked zone among the examples.
This is a decode/sanity check, **not** human ground-truth labeling or proof
that a map transition coincides with a visible regulatory sign. Map values
remain search hints only.

## Problems reproduced and fixed

1. **Schema mismatch:** importing the full openpilot LogReader selected an
   incompatible globally installed `car.capnp`. The isolated reader now loads
   explicit local log/car schemas, with a clear error if the car schema is absent.
2. **False exact startup matches:** segment 0 has 1180 decoded frames but
   `segmentId` 20 through 1199. Previously 1160 overlapping indices were marked
   exact despite the uncertain origin. The entire mismatched segment is now
   unresolved with `presentation_index_domain_mismatch`; no offset is guessed.
   Local logger source shows that startup can discard non-keyframes before
   recording while retaining encoder indices. The exact producer commit was
   unavailable locally, so that explanation is not treated as a verified join.
3. **No HEVC PTS:** valid frame durations now supply a separate media clock with
   explicit provenance, only for wholly timestamp-less raw HEVC. Missing or
   invalid durations preserve unknown media time.
4. **Empty successful clips:** input seeking in raw HEVC produced tiny MP4s with
   no video stream despite exit zero. Clip extraction now decodes before seeking
   and automatically reencodes raw HEVC. `extract-frames --start` uses the same
   seek ordering. Empty ffmpeg output is an error.
5. **Truncated Zstandard:** streaming decompression could return a complete
   message prefix without reporting an incomplete compressed frame. Every
   Zstandard frame now requires EOF; concatenated complete frames are supported.

## Verification

- Full suite with the optional parser interpreter: **154 passed**.
- `nnslr selftest`: **18 checks passed**.
- Regression tests cover shifted/missing index domains, unknown media clocks,
  mixed timestamps, real synthetic HEVC seeks and pixel content, empty outputs,
  schema layout, raw/bz2/zst Cap'n Proto round trips, truncated messages,
  truncated compressed frames and concatenated Zstandard frames.
- Final parser reread of all four real rlogs is identical to the validated
  metadata; the new compressed-EOF check accepts all four complete inputs.
- Import-isolation tests remain green. Optional parser and real ffmpeg tests
  skip when their external environment/tools are unavailable; all were enabled
  for the 154-test run reported here.
- Independent code review identified the Zstandard truncation issue above;
  its reproducing tests failed before the fix and pass afterward.

Reproduce the full optional parser tests with:

```sh
NNSLR_TEST_LOG_PYTHON=/path/to/parser-venv/bin/python python -m pytest -ra
nnslr selftest
```

## Remaining before T2 acceptance

- Establish the first-segment mapping from verified producer/packet evidence;
  do not simply subtract 20 based on this one recording.
- Complete the planned **ten-clip manual synchronization report**, with an
  explicit frame/capture reference and measured errors. Four center-image sanity
  checks and five candidate clips do not meet that criterion.
- Enrich extracted PNG manifests with source decoded index, geometry, capture
  time and alignment provenance. Existing discovery manifests do not contain
  those fields and must not silently be promoted to capture-aligned examples.
- Exercise partial GOPs, missing/truncated video and other camera streams on
  representative inputs. The current real run covers only narrow-camera HEVC.
- Proceed to T3 annotation/import/review/validation and leakage-safe splitting
  after those data contracts are settled. No model recognition or training is
  implemented by this increment.
<!-- [t2-validation] - END -->
