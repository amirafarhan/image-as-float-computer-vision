"""Streamlit demo: add noise to an image and compare denoising filters.

Run with:  streamlit run app.py
"""

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import image_filters as imf

st.set_page_config(page_title="Image Noise & Filters Lab", page_icon="🖼️", layout="wide")
MAX_SIDE = 1024
MAX_UPLOAD_MB = 10
DEFAULT_STRENGTH = {"gaussian": 0.01, "s&p": 0.05, "speckle": 0.04}


@st.cache_data
def load_sample(name: str) -> np.ndarray:
    return imf.SAMPLES[name]()


def load_upload(file) -> np.ndarray:
    if file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    img = Image.open(io.BytesIO(file.getvalue()))
    img = img.convert("RGB") if img.mode not in ("L", "RGB") else img
    if max(img.size) > MAX_SIDE:  # keep the app responsive
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img)


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(imf.to_uint8(img)).save(buf, format="PNG")
    return buf.getvalue()


st.title("🖼️ Image Noise & Filters Lab")
st.caption("Add Gaussian, salt & pepper or speckle noise to an image, then compare mean, median and Gaussian "
           "filters, measured with PSNR and SSIM against the clean image.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Image")
    source = st.radio("Source", ["Sample image", "Upload"], horizontal=True)
    raw = None
    try:
        if source == "Sample image":
            raw = load_sample(st.selectbox("Sample", list(imf.SAMPLES)))
        else:
            up = st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])
            if up is not None:
                raw = load_upload(up)
    except (UnidentifiedImageError, ValueError, OSError) as exc:
        st.error(f"Could not read the image: {exc}")
    grayscale = st.toggle("Convert to grayscale", value=True)

    st.header("2. Noise")
    noise_label = st.selectbox("Noise type", list(imf.NOISE_MODES))
    mode = imf.NOISE_MODES[noise_label]
    if mode == "s&p":
        strength = st.slider("Corrupted pixels (amount)", 0.01, 0.30, DEFAULT_STRENGTH[mode], 0.01)
    else:
        strength = st.slider("Variance", 0.001, 0.10, DEFAULT_STRENGTH[mode], 0.001, format="%.3f")
    seed = st.number_input("Random seed", 0, 9999, 0, help="Same seed = same noise, for fair comparisons")

    st.header("3. Filters")
    size = st.slider("Mean / median window (pixels)", 3, 15, 5, 2)
    sigma = st.slider("Gaussian sigma", 0.5, 5.0, 2.0, 0.5)

if raw is None:
    st.info("Upload an image in the sidebar to begin.")
    st.stop()

clean = imf.to_float(raw, grayscale=grayscale)
noisy = imf.add_noise(clean, mode, strength, int(seed))
filtered = {f: imf.apply_filter(noisy, f, size, sigma) for f in imf.FILTERS}
scores = pd.DataFrame([{"image": "Noisy", **imf.quality(clean, noisy)}] +
                      [{"image": f"{f} filter", **imf.quality(clean, out)} for f, out in filtered.items()])
best = scores.iloc[1:].sort_values("PSNR", ascending=False).iloc[0]

tab_view, tab_compare, tab_about = st.tabs(["🔬 Filter results", "📊 All noise × filter", "ℹ️ How it works"])

with tab_view:
    st.success(f"Best filter for this image and noise: **{best['image']}** — PSNR {best['PSNR']:.1f} dB, SSIM {best['SSIM']:.3f}")
    cols = st.columns(5)
    panels = [("Original", clean, None), (f"Noisy ({noise_label})", noisy, scores.iloc[0])] + [
        (f"{f} filter", out, scores.iloc[i + 1]) for i, (f, out) in enumerate(filtered.items())]
    for col, (title, img, sc) in zip(cols, panels):
        col.image(imf.to_uint8(img), caption=title if sc is None else f"{title} · {sc['PSNR']:.1f} dB · SSIM {sc['SSIM']:.2f}",
                  width="stretch", clamp=True)
    left, right = st.columns([2, 1])
    left.altair_chart(alt.Chart(scores).mark_bar().encode(
        x=alt.X("PSNR:Q", title="PSNR (dB, higher is better)"), y=alt.Y("image:N", sort="-x", title=None),
        color=alt.condition(alt.datum.image == best["image"], alt.value("#2a9d8f"), alt.value("#adb5bd")),
        tooltip=["image", alt.Tooltip("PSNR:Q", format=".2f"), alt.Tooltip("SSIM:Q", format=".3f")],
    ).properties(height=200), width="stretch")
    right.dataframe(scores.round({"PSNR": 2, "SSIM": 3}), hide_index=True, width="stretch")
    pick = right.selectbox("Download", ["Noisy"] + imf.FILTERS)
    right.download_button(f"Download {pick.lower()} image (PNG)", png_bytes(noisy if pick == "Noisy" else filtered[pick]),
                          f"{pick.lower()}.png", "image/png")

with tab_compare:
    st.caption("Every filter on every noise type, using the default strength of each noise and the current filter settings.")
    with st.spinner("Computing..."):
        rows = pd.DataFrame(imf.compare_all(clean, DEFAULT_STRENGTH, size, sigma, int(seed)))
    metric = st.radio("Metric", ["PSNR", "SSIM"], horizontal=True)
    heat = alt.Chart(rows).encode(x=alt.X("filter:N", sort=["None (noisy)"] + imf.FILTERS, title=None),
                                  y=alt.Y("noise:N", title=None))
    st.altair_chart((heat.mark_rect().encode(color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="viridis")))
                     + heat.mark_text(color="white", fontSize=14).encode(text=alt.Text(f"{metric}:Q", format=".2f" if metric == "PSNR" else ".3f")))
                    .properties(height=220), width="stretch")
    winners = rows[rows["filter"] != "None (noisy)"].loc[lambda d: d.groupby("noise")[metric].idxmax()]
    st.write("**Best filter per noise type:** " + ", ".join(f"{r.noise} → {r.filter}" for r in winners.itertuples()))

with tab_about:
    st.markdown("""
- **Noise** comes from `skimage.util.random_noise`:
  - *Gaussian* noise adds normally distributed values (the slider sets the variance);
  - *salt & pepper* sets a fraction of pixels to black or white;
  - *speckle* multiplies the image by noise.
- **Filters** come from `scipy.ndimage`:
  - the *mean* filter (`uniform_filter`) averages a window;
  - the *median* filter takes the window's median, which removes isolated outliers such as salt & pepper;
  - the *Gaussian* filter takes a weighted average with standard deviation *sigma*.
- **PSNR** (peak signal-to-noise ratio, in dB) and **SSIM** (structural similarity, from 0 to 1) compare each image with
  the clean original. Higher is better for both.
""")
