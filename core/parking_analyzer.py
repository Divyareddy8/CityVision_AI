import cv2
import numpy as np
from sklearn.cluster import KMeans

class ParkingAnalyzer:
    def __init__(self):
        self.parking_spots = {}
        self.occupied_spots = set()
        self.parking_occupancy = {}
        
    def detect_parking_spots(self, frame, method='grid'):
        if method == 'grid':
            return self.grid_based_detection(frame)
        elif method == 'contour':
            return self.contour_based_detection(frame)
        else:
            return self.learning_based_detection(frame)
    
    def grid_based_detection(self, frame, grid_size=(10, 10)):
        height, width = frame.shape[:2]
        cell_width = width // grid_size[0]
        cell_height = height // grid_size[1]
        
        spots = {}
        spot_id = 0
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x1 = i * cell_width
                y1 = j * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                spots[spot_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'occupied': False,
                    'confidence': 0.0
                }
                spot_id += 1
        
        self.parking_spots = spots
        return spots
    
    def contour_based_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spots = {}
        spot_id = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.8 < aspect_ratio < 1.2:
                    spots[spot_id] = {
                        'bbox': [x, y, x + w, y + h],
                        'occupied': False,
                        'confidence': 0.0
                    }
                    spot_id += 1
        
        self.parking_spots = spots
        return spots
    
    def analyze_parking_occupancy(self, objects, frame):
        vehicles = [obj for obj in objects if obj['class_name'] in ['car', 'bus', 'truck']]
        
        for spot_id, spot in self.parking_spots.items():
            spot_occupied = False
            max_iou = 0
            
            for vehicle in vehicles:
                iou = self.calculate_iou(spot['bbox'], vehicle['bbox'])
                if iou > max_iou:
                    max_iou = iou
                
                if iou > 0.3:
                    spot_occupied = True
            
            self.parking_spots[spot_id]['occupied'] = spot_occupied
            self.parking_spots[spot_id]['confidence'] = max_iou
        
        total_spots = len(self.parking_spots)
        occupied_spots = sum(1 for spot in self.parking_spots.values() if spot['occupied'])
        available_spots = total_spots - occupied_spots
        
        return {
            'total_spots': total_spots,
            'occupied_spots': occupied_spots,
            'available_spots': available_spots,
            'occupancy_rate': (occupied_spots / total_spots) * 100 if total_spots > 0 else 0,
            'spots_detail': self.parking_spots
        }
    
    def calculate_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0