# 🖼️ Image as Float — Noise & Denoising Filters

A computer-vision project on how images are represented as floating-point arrays, how noise corrupts them, and how
classic spatial filters remove it. Built with **scikit-image**, **SciPy** and **OpenCV**, plus an interactive
**Streamlit** lab for trying the filters on any image.

**🔗 Live demo:** https://image-as-float-computer-vision-ap3rjcnp8mvjwd8sbdw4pf.streamlit.app/

## 🚀 Features

- **Notebook (`image as float computer vision.ipynb`):**
  - load the scikit-image sample images; RGB-to-grayscale conversion;
  - convert to float with `img_as_float`;
  - add Gaussian, salt & pepper and speckle noise with `random_noise`;
  - denoise with SciPy's mean (`uniform_filter`), median and Gaussian filters.
- **Streamlit lab (`app.py`):**
  - use a sample image or upload a PNG/JPEG (up to 10 MB, resized to 1024 px), in colour or grayscale;
  - choose the noise type and strength, with a reproducible random seed;
  - set the filter window size and Gaussian sigma;
  - view the original, noisy and three filtered images side by side, each with PSNR and SSIM;
  - the best filter highlighted, with a PSNR chart;
  - a full noise × filter heatmap of PSNR or SSIM;
  - download any result as PNG.

## 📊 Results

Camera image in grayscale, filter window 5 px, Gaussian sigma 2. Each cell shows PSNR in dB / SSIM; higher is better.

| Noise | Noisy | Mean | Median | Gaussian |
|---|---|---|---|---|
| Gaussian (var 0.01) | 20.4 / 0.30 | 25.9 / 0.65 | **26.3** / 0.61 | 25.4 / **0.69** |
| Salt & pepper (5%) | 17.9 / 0.37 | 25.1 / 0.59 | **27.9 / 0.80** | 24.9 / 0.66 |
| Speckle (var 0.04) | 19.2 / 0.42 | 25.7 / 0.65 | **25.8** / 0.61 | 25.4 / **0.70** |

The median filter is clearly best for salt & pepper noise, because it ignores isolated extreme pixels. For Gaussian and
speckle noise the filters are close: the median filter gives the best PSNR and the Gaussian filter the best SSIM.

## 🛠️ Tech Stack

Python · NumPy · SciPy · scikit-image · OpenCV · Matplotlib · Pillow · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
image-as-float-computer-vision/
├── image as float computer vision.ipynb   # Notebook
├── image_filters.py                       # Noise, filters and quality metrics
├── app.py                                 # Streamlit lab
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/image-as-float-computer-vision.git
cd image-as-float-computer-vision
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Live demo: https://image-as-float-computer-vision-ap3rjcnp8mvjwd8sbdw4pf.streamlit.app/

Run the Streamlit lab locally (opens http://localhost:8501):

```bash
streamlit run app.py
```

Notebook: run `pip install jupyter matplotlib opencv-python`, then `jupyter notebook`.

## 👩‍💻 Author

Amira Farhan — Machine Learning & Computer Vision Engineer
