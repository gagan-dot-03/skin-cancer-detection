# Skin Cancer Detection using Deep Learning

## Overview

This project uses **EfficientNetB0 Transfer Learning** to classify skin lesion images into:

- Benign
- Malignant

The application is built using **Flask**, **TensorFlow**, and **OpenCV** and also generates a **Grad-CAM heatmap** to visualize the regions that influenced the model's prediction.

---

## Features

- Upload skin lesion image
- Automatic image preprocessing
- Deep learning prediction
- Benign / Malignant classification
- Confidence score
- Risk level
- Grad-CAM explainability
- Flask web interface
- Bootstrap 5 responsive design

---

## Technologies

- Python
- TensorFlow
- Keras
- Flask
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Folder Structure

```text
Skin-Cancer-Detection/
│
├── app.py
├── train.py
├── predict.py
├── gradcam.py
├── prepare_dataset.py
├── config.py
├── utils.py
├── requirements.txt
├── README.md
│
├── dataset/
├── models/
├── static/
└── templates/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Prepare Dataset

```bash
python prepare_dataset.py
```

---

## Train Model

```bash
python train.py
```

---

## Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Model

- EfficientNetB0
- Transfer Learning
- Binary Classification

---

## Future Improvements

- Multi-class skin lesion classification
- Cloud deployment
- Mobile application
- User authentication
- Database integration

---

## Author

P GAGAN KUMAR
