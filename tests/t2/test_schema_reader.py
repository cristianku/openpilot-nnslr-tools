# [schema-reader] - START
"""Optional parser integration: set NNSLR_TEST_LOG_PYTHON to its environment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nnslr_tools.comma_log import CommaLogError, read_log_metadata, encode_events

SCHEMA = '''@0xeaabf074c32aa734;
using Car = import "/car.capnp";
struct EncodeIndex {
  frameId @0 :UInt32;
  segmentNum @1 :Int32;
  segmentId @2 :UInt32;
  segmentIdEncode @3 :UInt32;
  timestampSof @4 :UInt64;
  timestampEof @5 :UInt64;
  type @6 :Text;
}
struct Event {
  logMonoTime @0 :UInt64;
  union {
    narrowRoadEncodeIdx @1 :EncodeIndex;
    unrelated @2 :Void;
  }
}
'''


@pytest.fixture
def schema_log(tmp_path):
    python = os.environ.get("NNSLR_TEST_LOG_PYTHON", sys.executable)
    check = subprocess.run([python, "-c", "import capnp, zstandard"], capture_output=True)
    if check.returncode:
        pytest.skip("optional pycapnp/zstandard parser environment not configured")
    cereal = tmp_path / "openpilot" / "cereal"
    car = tmp_path / "opendbc" / "car"
    cereal.mkdir(parents=True)
    car.mkdir(parents=True)
    (cereal / "log.capnp").write_text(SCHEMA)
    (car / "car.capnp").write_text("@0xdefaa074c32aa734;\nstruct Car {}\n")
    code = '''
import capnp, zstandard, bz2, sys
from pathlib import Path
root = Path(sys.argv[1])
capnp.remove_import_hook()
schema = capnp.load(str(root/'openpilot/cereal/log.capnp'), imports=[str(root/'opendbc/car')])
rows = []
for i in [1, 0]:
 m = schema.Event.new_message(logMonoTime=1000 + i)
 e = m.init('narrowRoadEncodeIdx')
 e.frameId = 40 + i
 e.segmentNum = 0
 e.segmentId = i
 e.segmentIdEncode = i
 e.timestampSof = 900 + i
 e.timestampEof = 950 + i
 e.type = 'fullHEVC'
 rows.append(m.to_bytes())
data = b''.join(rows)
(root/'rlog').write_bytes(data)
(root/'rlog.bz2').write_bytes(bz2.compress(data))
(root/'rlog.zst').write_bytes(zstandard.ZstdCompressor().compress(data))
(root/'truncated').write_bytes(data[:-1])
compressed = zstandard.ZstdCompressor(write_checksum=True).compress(data)
(root/'truncated.zst').write_bytes(compressed[:-1])
(root/'concatenated.zst').write_bytes(compressed + compressed)
(root/'partial-concatenated.zst').write_bytes(compressed + compressed[:-1])
'''
    subprocess.run([python, "-c", code, str(tmp_path)], capture_output=True, check=True)
    return tmp_path, python


@pytest.mark.parametrize("filename", ["rlog", "rlog.bz2", "rlog.zst"])
def test_schema_only_reader_real_serialization_and_compression(schema_log, filename):
    root, python = schema_log
    events = read_log_metadata(root / filename, openpilot_root=root, python_executable=python)
    assert [event.frame_id for event in events] == [40, 41]
    assert [event.timestamp_sof for event in events] == [900, 901]
    assert len(encode_events(events, service="narrowRoadEncodeIdx", segment_num=0)) == 2


@pytest.mark.parametrize("filename", ["truncated", "truncated.zst", "partial-concatenated.zst"])
def test_truncated_log_fails_without_returning_partial_metadata(schema_log, filename):
    root, python = schema_log
    with pytest.raises(CommaLogError, match="log_schema_reader_failed"):
        read_log_metadata(root / filename, openpilot_root=root, python_executable=python)
def test_concatenated_zstd_frames_are_all_read(schema_log):
    root, python = schema_log
    events = read_log_metadata(root / "concatenated.zst", openpilot_root=root, python_executable=python)
    assert [event.frame_id for event in events] == [40, 40, 41, 41]
# [schema-reader] - END
