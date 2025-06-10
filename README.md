# Brain_Tumor_Detection_Flask
This repository contains a web-based deep learning application for detecting brain tumors from medical images. The backend is built with Flask, and the model is served via Gunicorn. The detection models include YOLOv8 and DETR with a ResNet-101 backbone.

---
## Overview

The goal of this project is to provide an automated, lightweight, and accurate tool for identifying and classifying brain tumors in MRI scans.

The application supports detection of the following tumor types:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

---

## Installation
### 1. Clone the Repository

```bash
git clone https://github.com/yenyenn19/Brain_Tumor_Detection_Flask.git
cd Brain_Tumor_Detection_Flask
```

### 2. Add Model Files

```bash
Brain_Tumor_Detection_Flask/
└── model/
    ├── yolov8/
    │   └── best.pt              # YOLOv8 model
    └── detr_resnet_101/
        └── DETR_model           # DETR model file
```
⚠️ These files are not included in the repo due to size. Make sure they exist before building the image.

### 3. Build the Docker Image

```bash
docker build -t brain-tumor-detection .
```

### Run the Container

```bash
docker run -d -p 5000:5000 brain-tumor-detection
```
Then open your browser: http://localhost:5000

---
## Disclaimer
For Research Use Only. Not for use in diagnostic procedures.



