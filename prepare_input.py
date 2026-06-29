#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def read_bgr(source: Path) -> np.ndarray:
    """Decode an image to a BGR array.

    OpenCV handles png/jpg/jpeg/tiff/webp directly. For formats it lacks a
    codec for (e.g. JPEG XL, AVIF/HEIC), fall back to macOS `sips`, which
    transcodes via the system codecs to a temporary PNG that OpenCV can read.
    """
    image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if image is not None:
        return image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_png = Path(tmp) / "decoded.png"
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(source), "--out", str(tmp_png)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0 or not tmp_png.exists():
            raise RuntimeError(
                f"Could not read image (OpenCV + sips both failed): {source}\n"
                f"sips: {result.stderr.decode(errors='replace').strip()}"
            )
        image = cv2.imread(str(tmp_png), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise RuntimeError(f"Could not read image after sips transcode: {source}")
    return image


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    return np.where(
        image <= threshold,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def convert_image(source: Path, destination: Path) -> None:
    image = read_bgr(source)

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

    if not cv2.imwrite(
        str(destination),
        linear,
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF],
    ):
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
