# Installing IntrinsicHDR on Apple Silicon

The repository’s frozen `requirements.txt` should **not** be installed directly on macOS. It contains NVIDIA CUDA, cuDNN, NCCL, TensorRT, and `tensorflow-gpu` packages that cannot be installed natively on Apple Silicon. ([GitHub][4])

Use Miniforge and a controlled dependency set.

## 1. Install prerequisites

```bash
brew install miniforge git wget openexr
```

Initialize Conda:

```bash
conda init zsh
exec zsh
```

## 2. Clone the model

```bash
mkdir -p ~/Projects
cd ~/Projects

git clone https://github.com/compphoto/IntrinsicHDR.git
cd IntrinsicHDR
```

## 3. Create an ARM64 Python environment

The authors tested Python 3.9, so use that rather than the newest Python release. ([GitHub][2])

```bash
CONDA_SUBDIR=osx-arm64 conda create \
    --name intrinsic-hdr \
    python=3.9 \
    pip \
    -y

conda activate intrinsic-hdr
```

Confirm that the interpreter is native ARM64:

```bash
python -c 'import platform; print(platform.machine())'
```

Expected:

```text
arm64
```

## 4. Install PyTorch with Metal support

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    torch \
    torchvision
```

Verify MPS:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable")
PY
```

Expected:

```text
MPS built: True
MPS available: True
```

## 5. Install the non-CUDA dependencies

Start with:

```bash
python -m pip install \
    "numpy<2" \
    scipy \
    opencv-python \
    imageio \
    pillow \
    tqdm \
    matplotlib \
    scikit-image \
    kornia \
    timm \
    einops \
    geffnet \
    antialiased-cnns \
    pytorch-lightning==1.9.0 \
    lightning==1.9.5 \
    torchmetrics \
    hydra-core \
    colour \
    rawpy \
    OpenEXR \
    PyEXR
```

Then install the project itself without allowing its CUDA-heavy dependency list to run:

```bash
python -m pip install --no-deps -e .
```

## 6. Patch the device selection for MPS

The original line is:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Replace it with CUDA → MPS → CPU selection:

```bash
python - <<'PY'
from pathlib import Path

path = Path("inference.py")
text = path.read_text()

old = "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"

new = """DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)"""

if old not in text:
    raise SystemExit("Expected DEVICE line was not found; repository may have changed.")

path.write_text(text.replace(old, new, 1))
print("Patched inference.py for CUDA → MPS → CPU selection.")
PY
```

The model already transfers the three reconstruction networks to the selected device using `.to(device)`, so changing the global device selection should move those networks onto Metal. ([GitHub][5])

## 7. Enable CPU fallback for unsupported MPS operations

Some research models contain operations that PyTorch has not implemented on MPS. This environment variable permits those individual operations to execute on the CPU rather than aborting:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Persist it for this Conda environment:

```bash
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda deactivate
conda activate intrinsic-hdr
```

# Preparing the input

IntrinsicHDR’s main inference program expects **linear EXR input**, not an ordinary gamma-encoded JPEG or PNG. The official workflow first runs a TensorFlow 1-style dequantization and linearization network and then runs the PyTorch reconstruction stage. ([GitHub][2])

That TensorFlow preprocessing stage is the least Apple-Silicon-friendly part of the project. It uses `tensorflow.compat.v1`, graph sessions, and checkpoints from the older SingleHDR model. ([GitHub][6])

For digital art and CGI, a deterministic sRGB-to-linear conversion is a reasonable first experiment. It does not perform neural dequantization, but avoids installing the old TensorFlow stack.

## Create an sRGB-to-linear EXR converter

Create `prepare_input.py`:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    return np.where(
        image <= threshold,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def convert_image(source: Path, destination: Path) -> None:
    image = cv2.imread(
        str(source),
        cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH,
    )

    if image is None:
        raise RuntimeError(f"Could not read image: {source}")

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32)

    image = np.clip(image, 0.0, 1.0)

    # OpenCV reads BGR, but the sRGB transfer function applies equally
    # to each color component.
    linear = srgb_to_linear(image)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(destination), linear):
        raise RuntimeError(f"Could not write EXR: {destination}")

    print(f"{source} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an sRGB PNG/JPEG/TIFF into a linear float EXR."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    convert_image(args.input.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x prepare_input.py
```

Create directories:

```bash
mkdir -p inputs/linear outputs
```

Convert an image:

```bash
./prepare_input.py \
    "/path/to/source.png" \
    "inputs/linear/source.exr"
```

For JPEG:

```bash
./prepare_input.py \
    "/path/to/source.jpg" \
    "inputs/linear/source.exr"
```

# Running IntrinsicHDR

Run:

```bash
conda activate intrinsic-hdr
cd ~/Projects/IntrinsicHDR

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPENCV_IO_ENABLE_OPENEXR=1

python inference.py \
    --test_imgs inputs/linear \
    --output_path outputs \
    --use_exr
```

The reconstructed files should appear under:

```text
outputs/predictions/
```

The official program accepts a directory of EXR files, reconstructs each image, and writes HDR or EXR output. ([GitHub][5])

## Confirm Metal is selected

At startup, the patched script should print:

```text
mps
```

If it prints:

```text
cpu
```

check:

```bash
python - <<'PY'
import platform
import torch

print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```
