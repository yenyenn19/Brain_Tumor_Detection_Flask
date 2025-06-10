import os
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from brain_tumor_probs import preprocess_img, get_highest_probabilities_grids
from brain_tumor_visual import draw_grid_prediction
from detr_model import display_image


# Flask constructor
app = Flask(__name__)

# Folder to store uploaded files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the uploads folder
def init_files_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Clear out any existing files
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize the uploads folder
    init_files_folder()

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the selected model
    selected_model = request.form['model']

    # Save the file with a .jpg extension
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    class_probabilities_dict = {}
    score = 0
    # Load the selected model
    if selected_model == "yolov8":
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(BASE_DIR, 'model', 'yolov8', 'best.pt')
        model = YOLO(model_path)
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
        output_tensor = preprocess_img(img_path, model)

        highest_grids, highest_prob, highest_prob_grid_data, class_probabilities = get_highest_probabilities_grids(output_tensor)

        for grid_index, class_index, prob in highest_grids:
            res_img = draw_grid_prediction(img_path, highest_prob_grid_data, highest_prob, class_index, class_names=class_names)
    

        class_probabilities_dict = {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(class_probabilities)}
        # Convert the result image to a PIL Image and save it
        if isinstance(res_img, np.ndarray):
            res_img_pil = Image.fromarray(res_img)

            # Save the PIL image
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            res_img_pil.save(result_image_path)
        else:
            return jsonify({"error": "Prediction result is not valid."}), 500


    elif selected_model == "DETR-Resnet-101":
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(BASE_DIR, 'model', 'detr_resnet_101', 'DETR_model')
        
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        res_img, score, probs = display_image(img_path, model_path)

        # Map class names to their respective probabilities
        class_probabilities_dict = {class_names[i]: f"{probs[i]:.4f}" for i in range(len(class_names))}

        # Convert OpenCV image (BGR) to PIL Image (RGB)
        if isinstance(res_img, np.ndarray):
            res_img_pil = Image.fromarray(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
    
            # Save the PIL image
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            res_img_pil.save(result_image_path)

    # Sort dictionary by probabilities in descending order
    class_probabilities_dict = dict(sorted(class_probabilities_dict.items(), key=lambda item: item[1], reverse=True))

    # Save and get URLs for images
    original_image_url = url_for('static', filename='uploads/' + filename)
    result_image_url = url_for('static', filename='uploads/result_' + filename)

    # Return the prediction result page with the images
    return render_template("predict.html", 
                           original_image=original_image_url, 
                           result_image=result_image_url, 
                           class_probabilities=class_probabilities_dict)

# main driver function
if __name__ == '__main__':
    app.run(host="99.64.152.81", port=5000,debug=True)

