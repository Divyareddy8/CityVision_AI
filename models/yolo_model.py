import torch
import cv2
import numpy as np

class YOLOModel:
    def __init__(self, model_size='s', device='auto'):
        self.model_size = model_size
        self.device = device
        self.model = self.load_model()
        
    def load_model(self):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        
        model = torch.hub.load('ultralytics/yolov5', f'yolov5{self.model_size}', pretrained=True)
        model.to(device)
        return model
    
    def predict(self, image, confidence=0.5):
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()
        
        objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf >= confidence:
                objects.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
        
        return objects