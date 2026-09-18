# Proxmox GPU container setup

This guide prepares a Debian/Ubuntu Proxmox LXC/container for
`openpilot-nnslr-tools`, local comma video/log processing, and later V100
training.

## Assumptions

- Linux x86_64 container.
- NVIDIA device passthrough is already configured.
- `nvidia-smi` works **inside the container**.
- The NVIDIA driver is managed on the Proxmox host/container setup already.
- This guide does not install or modify the GPU driver.
- Commands below assume root. Remove `sudo`/adjust ownership if you use a
  dedicated user.

Check first:

```bash
nvidia-smi
python3 --version || true
uname -m
```

For two V100s, `nvidia-smi -L` should list both devices.

---

## 1. Base OS packages

On Debian/Ubuntu:

```bash
apt update
apt install -y \
  git \
  git-lfs \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  pkg-config \
  ca-certificates \
  curl \
  ffmpeg \
  zstd \
  jq
```

Verify the media tools:

```bash
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
zstd --version
```

---

## 2. Create software and data directories

Recommended layout:

```bash
mkdir -p /opt/nnslr
mkdir -p /srv/nnslr-data/{raw/routes,derived/probes,derived/alignment,derived/clips,derived/frames,manifests,annotations,datasets,splits,runs,evaluation}
```

The Git repository lives under `/opt/nnslr`.

All videos, logs, extracted frames, datasets, checkpoints and training output
live under `/srv/nnslr-data` and must not be committed to Git.

---

## 3. Clone NNSLR

```bash
cd /opt
git clone https://github.com/cristianku/openpilot-nnslr-tools.git nnslr
cd /opt/nnslr
```

For an existing checkout:

```bash
cd /opt/nnslr
git pull --ff-only
```

---

## 4. Create the NNSLR Python environment

```bash
cd /opt/nnslr

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```

Verify the CPU/tooling side:

```bash
nnslr --help
nnslr selftest
pytest -q
```

At this stage the core and T2 video tooling do not require PyTorch.

---

## 5. Configure the persistent data root

Create:

```bash
cat >/etc/profile.d/nnslr.sh <<'EOF'
export NNSLR_DATA_ROOT=/srv/nnslr-data
EOF

chmod 0644 /etc/profile.d/nnslr.sh
source /etc/profile.d/nnslr.sh
```

Check:

```bash
echo "$NNSLR_DATA_ROOT"
nnslr env
```

Expected data root:

```text
/srv/nnslr-data
```

---

## 6. Quick video test

Copy one local comma video into the data area, for example:

```text
/srv/nnslr-data/raw/routes/<route>/<segment>/fcamera.hevc
```

Then:

```bash
source /opt/nnslr/.venv/bin/activate

nnslr video-probe \
  /srv/nnslr-data/raw/routes/<route>/<segment>/fcamera.hevc
```

Broad discovery extraction defaults to approximately 1 fps:

```bash
nnslr extract-frames \
  --video /srv/nnslr-data/raw/routes/<route>/<segment>/fcamera.hevc \
  --output /srv/nnslr-data/derived/frames/discovery
```

Focused extraction around a known interval:

```bash
nnslr extract-frames \
  --video /srv/nnslr-data/raw/routes/<route>/<segment>/fcamera.hevc \
  --output /srv/nnslr-data/derived/frames/encounter-001 \
  --start 120 \
  --end 135 \
  --fps 10
```

No GPU is needed for these steps.

---

## 7. Local qlog/rlog parsing environment

NNSLR does not bundle openpilot/cereal. To parse real local `rlog/qlog`
files, point NNSLR at a local checkout matching the software that generated
the logs.

For Cristian's runtime fork:

```bash
cd /opt
git clone --recursive https://github.com/cristianku/nn-speed-limit-vision.git openpilot-src
```

If you already have the correct openpilot/sunnypilot source locally, use that
instead.

Current openpilot requires Python 3.12.x for its Python environment. A clean
way to keep this separate from the NNSLR venv is `uv`.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Prepare a Python 3.12 environment for the runtime source:

```bash
cd /opt/openpilot-src

uv python install 3.12
uv sync --frozen
```

Then add these variables:

```bash
cat >>/etc/profile.d/nnslr.sh <<'EOF'
export NNSLR_OPENPILOT_ROOT=/opt/openpilot-src
export NNSLR_OPENPILOT_PYTHON=/opt/openpilot-src/.venv/bin/python
EOF

source /etc/profile.d/nnslr.sh
```

Verify:

```bash
"$NNSLR_OPENPILOT_PYTHON" --version
test -f "$NNSLR_OPENPILOT_ROOT/openpilot/tools/lib/logreader.py" && echo OK
```

Now a local compressed log can be read directly:

```bash
source /opt/nnslr/.venv/bin/activate

nnslr log-metadata \
  /srv/nnslr-data/raw/routes/<route>/<segment>/rlog.zst \
  --output /srv/nnslr-data/manifests/log-metadata.jsonl
```

No manual `zstd -d` step is required when the matching openpilot LogReader
environment is available.

---

## 8. Video/log alignment

Example for the narrow road camera:

```bash
nnslr align-route \
  --video /srv/nnslr-data/raw/routes/<route>/<segment>/fcamera.hevc \
  --log /srv/nnslr-data/raw/routes/<route>/<segment>/rlog.zst \
  --stream narrow_road \
  --segment-num <segment> \
  --output /srv/nnslr-data/derived/alignment/<segment>.jsonl
```

Inspect:

```bash
nnslr alignment-report \
  /srv/nnslr-data/derived/alignment/<segment>.jsonl
```

The production alignment uses openpilot `EncodeIndex.segmentId`, which is the
camera-file index in presentation order. It does not equate encode order with
decoded presentation order.

---

## 9. Prepare the V100 PyTorch environment

The Tesla V100 is Volta, compute capability `sm_70`.

Do **not** install an arbitrary current PyTorch CUDA wheel. Newer CUDA 13.x
binary builds do not support Volta. For the V100, use the CUDA 12.6 PyTorch
build.

Keep training dependencies in a separate venv:

```bash
cd /opt/nnslr

python3 -m venv .venv-gpu
source .venv-gpu/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.14.0 \
  torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -e .
```

Why pin it:

- V100 = Volta / `sm_70`.
- PyTorch CUDA 12.6 binaries still include Volta support.
- CUDA 13.x binaries do not.
- PyTorch 2.14 is the last release line with published CUDA 12.6 wheels for
  Volta; later releases require staying on this version or building PyTorch
  from source for `sm_70`.

Do not install a separate CUDA toolkit merely because `nvidia-smi` works.
The PyTorch wheel ships its CUDA runtime dependencies; the host/container
NVIDIA driver must only be compatible with the selected runtime.

---

## 10. Verify both V100s from PyTorch

With `.venv-gpu` active:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("compiled arch list:", torch.cuda.get_arch_list())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        i,
        p.name,
        f"{p.total_memory / 1024**3:.1f} GiB",
        "capability",
        torch.cuda.get_device_capability(i),
    )

x = torch.randn(4096, 4096, device="cuda:0", dtype=torch.float16)
y = x @ x
print("GPU0 FP16 smoke:", float(y[0, 0]))

if torch.cuda.device_count() > 1:
    x1 = torch.randn(2048, 2048, device="cuda:1", dtype=torch.float16)
    y1 = x1 @ x1
    print("GPU1 FP16 smoke:", float(y1[0, 0]))
PY
```

For V100 you want to see:

```text
CUDA available: True
GPU count: 2
capability (7, 0)
```

and `sm_70` should be present in the compiled architecture list.

Also check:

```bash
nvidia-smi
```

---

## 11. Recommended shell workflow

For video/dataset work:

```bash
source /opt/nnslr/.venv/bin/activate
source /etc/profile.d/nnslr.sh
```

For GPU training later:

```bash
source /opt/nnslr/.venv-gpu/bin/activate
source /etc/profile.d/nnslr.sh
```

Do not mix the two virtual environments.

---

## 12. Updating NNSLR

CPU/T2 environment:

```bash
cd /opt/nnslr
git pull --ff-only

source .venv/bin/activate
python -m pip install -e .
pytest -q
```

GPU environment after training dependencies change:

```bash
cd /opt/nnslr
source .venv-gpu/bin/activate
python -m pip install -e .
```

---

## 13. Basic troubleshooting

### `nvidia-smi` works but PyTorch says CUDA is unavailable

Check:

```bash
source /opt/nnslr/.venv-gpu/bin/activate

python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import torch; print(torch.cuda.get_arch_list())'
nvidia-smi
```

On V100, if the installed wheel does not contain `sm_70`, replace it with the
CUDA 12.6 build documented above.

### `ffmpeg` or `ffprobe` not found

```bash
apt install -y ffmpeg
which ffmpeg
which ffprobe
```

### qlog/rlog parser cannot import openpilot

Check:

```bash
echo "$NNSLR_OPENPILOT_ROOT"
echo "$NNSLR_OPENPILOT_PYTHON"

"$NNSLR_OPENPILOT_PYTHON" - <<'PY'
from openpilot.tools.lib.logreader import LogReader
print("LogReader import OK")
PY
```

If this fails, rebuild/sync the separate openpilot environment, not the NNSLR
core environment.

### Keep raw data out of Git

Before doing any development from the container:

```bash
cd /opt/nnslr
git status
```

Videos, logs, datasets, runs and checkpoints belong under
`/srv/nnslr-data`, not under the repository.
