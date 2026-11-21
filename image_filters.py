"""Noise and denoising-filter helpers from ``image as float computer vision.ipynb``.

Same operations as the notebook: images converted with ``img_as_float``, noise from
``skimage.util.random_noise`` (gaussian, s&p, speckle) and denoising with SciPy's
``uniform_filter`` (mean), ``median_filter`` and ``gaussian_filter``. Colour images are
filtered per channel. Quality is measured against the clean image with PSNR and SSIM.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise

SAMPLES = {"Camera man": data.camera, "Astronaut": data.astronaut, "Cat (Chelsea)": data.chelsea, "Coffee": data.coffee}
NOISE_MODES = {"Gaussian": "gaussian", "Salt & pepper": "s&p", "Speckle": "speckle"}
FILTERS = ["Mean", "Median", "Gaussian"]


def to_float(img: np.ndarray, grayscale: bool = False) -> np.ndarray:
    """Float image in [0, 1]; drops alpha; optional RGB -> gray (correct RGB weights)."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img = img_as_float(img)
    if grayscale and img.ndim == 3:
        img = rgb2gray(img)
    return img


def add_noise(img: np.ndarray, mode: str, strength: float, seed: int = 0) -> np.ndarray:
    """`strength` is the variance for gaussian/speckle and the corrupted fraction for s&p."""
    if mode not in NOISE_MODES.values():
        raise ValueError(f"Unknown noise mode: {mode}")
    kwargs = {"amount": strength} if mode == "s&p" else {"var": strength}
    return random_noise(img, mode=mode, rng=seed, **kwargs)


def apply_filter(img: np.ndarray, name: str, size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """Notebook filters; spatial only, so colour channels are not mixed."""
    spatial = (size, size) + ((1,) if img.ndim == 3 else ())
    if name == "Mean":
        return uniform_filter(img, size=spatial)
    if name == "Median":
        return median_filter(img, size=spatial)
    if name == "Gaussian":
        return gaussian_filter(img, sigma=(sigma, sigma) + ((0,) if img.ndim == 3 else ()))
    raise ValueError(f"Unknown filter: {name}")


def quality(clean: np.ndarray, test: np.ndarray) -> dict:
    kw = {"channel_axis": -1} if clean.ndim == 3 else {}
    return {"PSNR": float(peak_signal_noise_ratio(clean, test, data_range=1.0)),
            "SSIM": float(structural_similarity(clean, test, data_range=1.0, **kw))}


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def compare_all(clean: np.ndarray, strengths: dict, size: int = 5, sigma: float = 2.0, seed: int = 0) -> list[dict]:
    """PSNR/SSIM of every filter on every noise type (the notebook experiment, measured)."""
    rows = []
    for label, mode in NOISE_MODES.items():
        noisy = add_noise(clean, mode, strengths[mode], seed)
        rows.append({"noise": label, "filter": "None (noisy)", **quality(clean, noisy)})
        for f in FILTERS:
            rows.append({"noise": label, "filter": f, **quality(clean, apply_filter(noisy, f, size, sigma))})
    return rows
