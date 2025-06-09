# 1. Start with a lightweight Python 3.9 image
FROM python:3.11-slim

# 2. Set environment variables to improve performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system libraries needed by OpenCV and others
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your local files into the container
COPY . /app

# 6. Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# 7. Expose port 5000 (Flask runs here)
EXPOSE 5000

# 8. Start the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
