🖼️ Image as Float – Computer Vision Project

This repository contains a Jupyter Notebook demonstrating how to convert and process images as floating-point arrays for computer vision tasks using Python, OpenCV, and NumPy.

It focuses on understanding how images are represented numerically in machine learning pipelines and how converting images to float improves preprocessing, normalization, and model performance.

📌 Project Overview

In computer vision, images are not just visual data — they are matrices of pixel values. This project explores:

Loading images using OpenCV
Understanding pixel representation (RGB channels)
Converting images to normalized floating-point format
Comparing integer vs float image representations
Preparing images for machine learning and deep learning pipelines
🧠 Key Concepts Covered
Digital image representation (pixels as arrays)
RGB color channels
NumPy array manipulation
Image normalization
Floating-point conversion using skimage.img_as_float
Basic preprocessing for computer vision models
🛠️ Technologies Used
Python 🐍
OpenCV (cv2)
NumPy
Matplotlib
scikit-image
📂 Notebook

The main notebook in this repository:

image as float computer vision.ipynb

It includes step-by-step explanations with code examples for:

Reading images
Displaying pixel values
Converting images to float format
Understanding the effect of normalization
🚀 How to Run
Clone the repository:
git clone https://github.com/amirafarhan/image-as-float-computer-vision.git
Install dependencies:
pip install opencv-python numpy matplotlib scikit-image
Run Jupyter Notebook:
jupyter notebook
Open:
image as float computer vision.ipynb
📊 Why This Matters

Converting images to float is a key step in:

Deep Learning models (CNNs)
Image normalization
Improving training stability
Standardizing pixel intensity values
📌 Example
from skimage import img_as_float
import cv2

image = cv2.imread("image.jpg")
float_image = img_as_float(image)

print(image[100, 100])       # integer pixel values
print(float_image[100, 100]) # normalized float values (0 → 1)
📈 Future Improvements
Add image augmentation examples
Integrate with CNN training pipeline
Add visualization comparisons (before/after normalization)
Support batch image processing
👨‍💻 Author

Amira Farhan

Computer Vision & Deep Learning Enthusiast
