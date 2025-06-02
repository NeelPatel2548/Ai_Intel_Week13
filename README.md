# InspectorsAlly: Anomaly Detector

A Streamlit-based web application that uses machine learning to detect defects in product images.

## Features

- Upload and analyze product images
- Real-time defect detection
- Confidence score for predictions
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NeelPatel2548/toothbrush_Week13.git
cd toothbrush_Week13
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Model

This application uses a TensorFlow model trained on Teachable Machine to detect defects in product images. The model is stored in `keras_model.h5`.

## Requirements

- Python 3.x
- Streamlit
- TensorFlow
- NumPy
- Pillow 