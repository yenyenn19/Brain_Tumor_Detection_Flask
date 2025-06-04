import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import torch.nn.functional as F


def display_image(image_path, model_path):
    CONFIDENCE_THRESHOLD = 0.35
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id2label = {
    0: 'tumor-vR5S',
    1: 'Glioma',
    2: 'Meningioma',
    3: 'No Tumor',
    4: 'Pituitary'
}
    image_preprocessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    image = cv2.imread(image_path)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    torch.set_float32_matmul_precision('medium')
    # Preprocess image and run model inference
    inputs = image_preprocessor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process to extract bounding boxes, scores, and labels
    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = image_preprocessor.post_process_object_detection(
        outputs=outputs, threshold=0.5, target_sizes=target_sizes  # Set threshold to 0.0 to keep all detections
    )[0]

    for box, score, label in zip(results['boxes'].cpu().numpy(), results['scores'].cpu().numpy(), results['labels'].cpu().numpy()):
        if score >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label_text = f"{id2label[label]}: {score:.4f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Get the logits (raw scores before applying softmax) from the model's outputs
    logits = outputs.logits  # Shape: [batch_size, num_detections, num_classes]

    # Identify the detection with the highest confidence score
    probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    max_detection_idx = torch.argmax(probs[..., :-1].max(dim=-1).values)  # Ignore the background class if last

    # Extract probabilities for all classes from this highest-confidence detection
    highest_probs = probs[0, max_detection_idx]  # Extract for the relevant detection grid

    # Convert to a Python list if needed for easier inspection
    highest_probs = highest_probs.tolist()
    selected_probs = highest_probs[1:5]

    print("Probabilities for all classes in the highest-confidence detection grid:", selected_probs)

    return image, score, selected_probs
