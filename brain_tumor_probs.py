import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from brain_tumor_visual import draw_grid_prediction

def preprocess_img(img_path, model):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to YOLOv8 input size
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference and capture the raw output
    with torch.no_grad():
        output = model.model(image_tensor)  # Run the model inference
        output0_tensor = output[0]  # Access the output0 tensor
        return output0_tensor

def get_highest_probabilities_grids(predOutput, m_labels=4):
    # Determine if YOLOv8 architecture is used based on the grid size
    m_grids = predOutput.shape[2]
    yolov8 = m_grids == 8400

    # Initialize variables to store the highest probability and corresponding grid
    highest_prob = -1
    highest_prob_grids = []
    highest_prob_grid_data = None
    class_probabilities = []

    # Iterate through the grids
    for i in range(m_grids):
        for j in range(m_labels):
            prob = predOutput[0, j + 4, i]  # Class probabilities start from index 4

            # Check if this probability is higher than the current highest
            if prob > highest_prob:
                highest_prob = prob
                highest_prob_grids = [(i, j, prob)]  # Reset the list with the new highest probability grid
                highest_prob_grid_data = predOutput[0, :4, i]
                class_probabilities = predOutput[0, 4:4 + m_labels, i].tolist()  # Get all class probabilities for this grid
            elif prob == highest_prob:
                highest_prob_grids.append((i, j, prob))  # Add grid if it ties with the highest probability

    return highest_prob_grids, highest_prob, highest_prob_grid_data, class_probabilities
