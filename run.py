import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import torch
from ultralytics import YOLO
import warnings
import logging
import pandas as pd


logging.basicConfig(
    level=logging.INFO+10,  # Change to DEBUG for more verbosity
    format='[%(levelname)s] %(message)s'
)
#level=logging.CRITICAL + 1,  # Change to DEBUG for more verbosity
max_images = 10000000000


warnings.filterwarnings('ignore')
# Configuration
E_WASTE_DIR = 'eWaste'
OTHER_DIR = 'other'
# E_WASTE_CLASSES = {62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 78, 79}  # COCO class indices for e-waste
E_WASTE_CLASSES = {64,66,67,68}
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45
# REDUCED_FP_CONF = 0.7
# REDUCED_FP_IOU = 0.6

REDUCED_FN_CONF = 0.25   #(lower confidence threshold, so we accept more detections)
REDUCED_FN_IOU = 0.2

with open("coco.names", "r") as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

# Load YOLO models
def load_yolo_models():
    models = {
        'yolov3': cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg'),
        'yolov4': cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg'),
        'yolov5': YOLO('yolov5s.pt'),
        'yolov8': YOLO('yolov8n.pt')
    }
    return models
# Get COCO class names
def get_coco_classes():
    with open('coco.names', 'r') as f:
        return [line.strip() for line in f.readlines()]
# Process image with YOLOv3/v4 (OpenCV)
def process_opencv_yolo(net, image_path, conf_thres, iou_thres):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image not found: {image_path}")
        return 0
        
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #if debug:
        # print(f"Output layers: {output_layers}")
    outputs = net.forward(output_layers)
    logging.debug(f"Number of output layers: {len(outputs)}")
    boxes, confidences, class_ids = [], [], []
    h, w = image.shape[:2]

    detection_count = 0
    for output in outputs:
        logging.debug(f"Processing output a layer of {len(output)} detections")
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thres:
                detection_count += 1
                center_x, center_y, width, height = detection[:4] * np.array([w, h, w, h])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                logging.info(f"[DETECTION] Class name(ID): {COCO_CLASSES[class_id]}({class_id}), Confidence: {confidence:.2f}, Box: ({x}, {y}, {width}, {height})")
            #else:
                #logging.info(f"[SKIPPED] Low confidence ({confidence:.2f}) below threshold for class {class_id}")

    if detection_count == 0:
        logging.debug("No detections above confidence threshold.")

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    logging.debug(f"{len(indices)} detections remaining after NMS")

    for i in indices:
        class_id = class_ids[i]
        
        if class_id in E_WASTE_CLASSES:
            logging.info(f"[RESULT - EWa] Final detection: Class ID {class_id}")
   
            return image_path, 1, 0.5, class_id, COCO_CLASSES[class_id]
        else:
            logging.info(f"[RESULT - Not] Final detection: Class ID {class_id}")
    return image_path, 0, 0.5, class_id, COCO_CLASSES[class_id]
# Process image with YOLOv5/v8 (Ultralytics)
def process_ultralytics_yolo(model, image_path, conf_thres, iou_thres):
    results = model(image_path, conf=conf_thres, iou=iou_thres, verbose=False)
    max_conf = -1
    max_cls = None

    for result in results:
        #print (results)
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            #print(f"-- {cls} ---")
            if(conf.item()>max_conf):
                max_conf=conf.item()
                max_cls=int(cls)
            if int(cls) in E_WASTE_CLASSES:
                logging.info(f"[RESULT - EWa] Final detection: Class ID {int(cls)}")
                return image_path, 1, conf.item(), int(cls), COCO_CLASSES[int(cls)]
    logging.info(f"[RESULT - Not] Final detection: Class ID {max_cls}")                
    if(max_cls==None):
        max_cls=80
    return image_path, 0, max_conf, max_cls, COCO_CLASSES[int(max_cls)]

# Get all image paths and true labels
def get_image_data():
    image_paths, true_labels = [], []
    for dir_name, label in [(E_WASTE_DIR, 1), (OTHER_DIR, 0)]:
        count=0
        for root, _, files in os.walk(dir_name):
            for filename in files:
                if count >= max_images:
                    break
                count += 1
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, filename))
                    true_labels.append(label)
    return image_paths, true_labels

def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    myMet = (0.1 * fp + 0.9 * fn) / (tp + tn + fp + fn) 
    return tp, fp, tn, fn, accuracy, precision, recall, myMet


# Main processing function


def main():
    # Load models and class names
    models = load_yolo_models()
    results = []

    image_paths, true_labels = get_image_data()
    # Initialize predictions storage
    predictions = defaultdict(list)
    model_configs = {
        'yolov3_default': (models['yolov3'], DEFAULT_CONF, DEFAULT_IOU, 'opencv'),
        'yolov3_reduced_fn': (models['yolov3'], REDUCED_FN_CONF, REDUCED_FN_IOU, 'opencv'),
        'yolov4_default': (models['yolov4'], DEFAULT_CONF, DEFAULT_IOU, 'opencv'),
        'yolov4_reduced_fn': (models['yolov4'], REDUCED_FN_CONF, REDUCED_FN_IOU, 'opencv'),
        'yolov5_default': (models['yolov5'], DEFAULT_CONF, DEFAULT_IOU, 'ultralytics'),
        'yolov5_reduced_fn': (models['yolov5'], REDUCED_FN_CONF, REDUCED_FN_IOU, 'ultralytics'),
        'yolov8_default': (models['yolov8'], DEFAULT_CONF, DEFAULT_IOU, 'ultralytics'),
        'yolov8_reduced_fn': (models['yolov8'], REDUCED_FN_CONF, REDUCED_FN_IOU, 'ultralytics')
    }
    # Process each image with each model configuration
    img_count = 0
    for img_path in image_paths:
        logging.info(f"Processing {img_path}")
        for config_name, (model, conf, iou, model_type) in model_configs.items():
            if model_type == 'opencv':
                full_path,pred,confidence,classId, className = process_opencv_yolo(model, img_path, conf, iou)
                results.append({"model":config_name,"filename": full_path,"prediction": pred,"confidence": confidence, "classID" : classId, "class" : className
            })     

            else:
                full_path,pred,confidence,classId, className = process_ultralytics_yolo(model, img_path, conf, iou)
                results.append({"model":config_name,"filename": full_path,"prediction": pred,"confidence": confidence, "classID" : classId, "class" : className
            }) 
            predictions[config_name].append(pred)
            # logging.info(f"Predicted {pred} for {img_path} with {config_name}")
            img_count=img_count+1
            print(f"Processing file: {img_path} ({img_count}/{len(image_paths)*len(model_configs)})              ", end='\r', flush=True)

    df = pd.DataFrame(results)
    df.to_csv("results_redFN.csv", index=False)


    cMatrix = []


    # Compute confusion matrices for individual models
    for config_name in model_configs:
        cm = confusion_matrix(true_labels, predictions[config_name])
        tp, fp, tn, fn, accuracy, precision, recall , myMet = calculate_metrics(cm)
        cMatrix.append({
            'Model': config_name,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'myMet':myMet
        })
    
        print(f"\nConfusion Matrix for {config_name}:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    # Ensemble predictions using majority voting


    # Ensemble 1: All models with 5/8 vote
    ensemble_5_8 = []
    for i in range(len(image_paths)):
        votes = [predictions[config][i] for config in model_configs]
        ensemble_5_8.append(1 if sum(votes) >= 5 else 0)
    

    # Ensemble 2: All models with 3/8 vote
    ensemble_3_8 = []
    for i in range(len(image_paths)):
        votes = [predictions[config][i] for config in model_configs]
        ensemble_3_8.append(1 if sum(votes) >= 3 else 0)
    
    # Ensemble 3: Only default models with 2/4 vote
    default_configs = [config for config in model_configs if 'default' in config]
    ensemble_default_2_4 = []
    for i in range(len(image_paths)):
        votes = [predictions[config][i] for config in default_configs]
        ensemble_default_2_4.append(1 if sum(votes) >= 2 else 0)
    
    # Ensemble 4: Only reduced FN models with 2/4 vote
    reduced_configs = [config for config in model_configs if 'reduced_fn' in config]
    ensemble_reduced_2_4 = []
    for i in range(len(image_paths)):
        votes = [predictions[config][i] for config in reduced_configs]
        ensemble_reduced_2_4.append(1 if sum(votes) >= 2 else 0)
    
    # Compute confusion matrices for all ensembles
    ensembles = {
        'Ensemble (All models, 5/8 vote)': ensemble_5_8,
        'Ensemble (All models, 3/8 vote)': ensemble_3_8,
        'Ensemble (Default models, 2/4 vote)': ensemble_default_2_4,
        'Ensemble (Reduced FN models, 2/4 vote)': ensemble_reduced_2_4
    }
    


    for name, preds in ensembles.items():
        cm = confusion_matrix(true_labels, preds)
        tp, fp, tn, fn, accuracy, precision, recall, myMet = calculate_metrics(cm)
        cMatrix.append({
            'Model': name,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'myMet': myMet
        })
        print(f"\nConfusion Matrix for {name}:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")


    df = pd.DataFrame(cMatrix)
    df.to_csv("cMatrix_redFN.csv", index=False)



if __name__ == "__main__":
    main()