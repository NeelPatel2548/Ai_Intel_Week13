# InspectorsAlly: Anomaly Detector

A Streamlit-based web application for detecting defects in products using computer vision and machine learning.

## Features

- **Image Upload**: Upload product images for defect detection
- **Live Camera**: Real-time defect detection using camera feed
- **Image Cropping**: Crop images to focus on specific areas
- **Real-time Analysis**: Instant prediction results with confidence scores
- **Responsive UI**: Clean and intuitive user interface

## Requirements

- Python 3.7+
- Streamlit
- Keras
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd InspectorsAlly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model file (`keras_model.h5`) in the project directory

## Usage

Run the application:
```bash
streamlit run streamlit_app.py
```

## Features in Detail

### Image Upload
- Supports JPG, JPEG, and PNG formats
- Automatic image resizing and optimization
- Instant defect detection

### Live Camera
- Real-time camera feed
- Image cropping capabilities
- Instant analysis of captured frames

### Analysis Results
- Defect classification (Defective/Non-Defective)
- Confidence score display
- Visual preview of analyzed images

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 