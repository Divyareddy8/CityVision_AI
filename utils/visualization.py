import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Visualization:
    def __init__(self):
        self.colors = {
            'car': (0, 255, 0),
            'person': (255, 0, 0),
            'bus': (0, 0, 255),
            'truck': (255, 255, 0),
            'motorcycle': (0, 255, 255)
        }
    
    def draw_detections(self, image, objects, show_labels=True):
        result_image = image.copy()
        
        for obj in objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if show_labels:
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                cv2.rectangle(result_image, 
                            (bbox[0], bbox[1] - label_size[1] - 10),
                            (bbox[0] + label_size[0], bbox[1]), 
                            color, -1)
                
                cv2.putText(result_image, label, 
                          (bbox[0], bbox[1] - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def draw_traffic_analysis(self, image, analysis):
        result_image = image.copy()
        height, width = image.shape[:2]
        
        congestion_level = analysis['congestion_level']
        vehicle_count = analysis['total_vehicles']
        
        if congestion_level == 'Low':
            color = (0, 255, 0)
        elif congestion_level == 'Medium':
            color = (0, 255, 255)
        elif congestion_level == 'High':
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(result_image, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.putText(result_image, f"Congestion: {congestion_level}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result_image, f"Vehicles: {vehicle_count}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, f"Density: {analysis['traffic_density']:.1f}%", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def draw_crowd_density(self, image, analysis):
        result_image = image.copy()
        
        total_people = analysis['total_people']
        density_level = analysis['density_level']
        
        if density_level == 'Very Low':
            color = (0, 255, 0)
        elif density_level == 'Low':
            color = (0, 255, 255)
        elif density_level == 'Medium':
            color = (0, 165, 255)
        elif density_level == 'High':
            color = (0, 0, 255)
        else:
            color = (128, 0, 128)
        
        cv2.rectangle(result_image, (10, 10), (300, 90), (0, 0, 0), -1)
        cv2.putText(result_image, f"Crowd: {density_level}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result_image, f"People: {total_people}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def create_heatmap(self, points, image_shape, radius=20):
        heatmap = np.zeros(image_shape[:2], dtype=np.float32)
        
        for point in points:
            cv2.circle(heatmap, (int(point[0]), int(point[1])), radius, 1, -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_colored