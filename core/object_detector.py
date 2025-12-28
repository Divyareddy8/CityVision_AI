import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import time

class ObjectDetector:
    def __init__(self, model_type='yolov5', model_path=None, confidence_threshold=0.5):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = self.load_model(model_path)
        self.class_names = self.get_class_names()
        
    def load_model(self, model_path):
        if self.model_type == 'yolov5':
            if model_path:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            else:
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif self.model_type == 'faster_rcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
        else:
            raise ValueError("Unsupported model type")
        return model
    
    def get_class_names(self):
        if self.model_type == 'yolov5':
            return self.model.names
        else:
            return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported image format")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def detect_objects(self, image):
        preprocessed_image = self.preprocess_image(image)
        
        if self.model_type == 'yolov5':
            results = self.model(preprocessed_image)
            detections = results.xyxy[0].cpu().numpy()
            
            objects = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.confidence_threshold:
                    objects.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.class_names[int(cls)]
                    })
        else:
            image_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                predictions = self.model([image_tensor])
            
            objects = []
            for i in range(len(predictions[0]['boxes'])):
                if predictions[0]['scores'][i] >= self.confidence_threshold:
                    bbox = predictions[0]['boxes'][i].cpu().numpy().astype(int)
                    objects.append({
                        'bbox': bbox.tolist(),
                        'confidence': float(predictions[0]['scores'][i]),
                        'class_id': int(predictions[0]['labels'][i]),
                        'class_name': self.class_names[int(predictions[0]['labels'][i])]
                    })
        
        return objects
    
    def filter_urban_objects(self, objects, target_classes=None):
        if target_classes is None:
            target_classes = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle']
        
        filtered_objects = []
        for obj in objects:
            if obj['class_name'] in target_classes:
                filtered_objects.append(obj)
        
        return filtered_objects