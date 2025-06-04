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

### 2. Set Up Python Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---
## Running the App with Gunicorn
If virtual environment is activated
  
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

OR run directly without activating the venv
```bash
./myenv/bin/gunicorn --bind 0.0.0.0:5000 app:app
```

---
## Disclaimer
For Research Use Only. Not for use in diagnostic procedures.



