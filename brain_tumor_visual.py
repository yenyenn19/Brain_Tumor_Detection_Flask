import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

def denormalize_bbox(bbox, img_width, img_height):

    x_center, y_center, width, height = bbox
    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)

    else:
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2))
        y_min = int((y_center - height / 2))
        x_max = int((x_center + width / 2))
        y_max = int((y_center + height / 2))
    
    return x_min, y_min, x_max, y_max

def draw_bounding_box(image, bbox, highest_prob, class_id, class_names, color=(0, 255, 0), thickness=2):

    x_min, y_min, x_max, y_max = bbox
    
    # Draw the rectangle on the image
    image_with_bbox = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # Add the class label text to the bounding box
    label = f"{class_names[class_id]}: {highest_prob: .2f}"
    cv2.putText(image_with_bbox, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_with_bbox


def draw_grid_prediction(img_path, grid_bbox, highest_prob, class_id, class_names, color=(255, 0, 0), thickness=2):

    image = cv2.imread(img_path)
    img_height, img_width, _ = image.shape
    
    # Denormalize the bounding box from grid prediction
    denorm_bbox = denormalize_bbox(grid_bbox, img_width, img_height)
    
    # Draw the bounding box using the existing draw_bounding_box function
    image_with_bbox = draw_bounding_box(image, denorm_bbox, highest_prob, class_id, class_names, color, thickness)
    
    return image_with_bbox
