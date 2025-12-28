import cv2
import numpy as np
from collections import deque, defaultdict
import time

class TrafficAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.vehicle_count = defaultdict(int)
        self.traffic_flow = deque(maxlen=100)
        self.speed_estimates = {}
        self.trajectories = defaultdict(list)
        self.frame_count = 0
        
    def analyze_traffic_flow(self, objects, frame_shape):
        current_vehicles = self.filter_vehicles(objects)
        self.update_vehicle_count(current_vehicles)
        
        traffic_density = self.calculate_traffic_density(current_vehicles, frame_shape)
        congestion_level = self.assess_congestion(traffic_density)
        
        analysis = {
            'vehicle_count': dict(self.vehicle_count),
            'total_vehicles': len(current_vehicles),
            'traffic_density': traffic_density,
            'congestion_level': congestion_level,
            'timestamp': time.time()
        }
        
        self.traffic_flow.append(analysis)
        return analysis
    
    def filter_vehicles(self, objects):
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
        return [obj for obj in objects if obj['class_name'] in vehicle_classes]
    
    def update_vehicle_count(self, vehicles):
        for vehicle in vehicles:
            vehicle_type = vehicle['class_name']
            self.vehicle_count[vehicle_type] += 1
    
    def calculate_traffic_density(self, vehicles, frame_shape):
        if not vehicles:
            return 0.0
        
        frame_area = frame_shape[0] * frame_shape[1]
        total_vehicle_area = 0
        
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_vehicle_area += bbox_area
        
        density = (total_vehicle_area / frame_area) * 100
        return min(density, 100.0)
    
    def assess_congestion(self, density):
        if density < 10:
            return 'Low'
        elif density < 30:
            return 'Medium'
        elif density < 50:
            return 'High'
        else:
            return 'Severe'
    
    def estimate_speed(self, current_objects, previous_objects, fps):
        speeds = []
        
        for curr_obj in current_objects:
            for prev_obj in previous_objects:
                if self.is_same_vehicle(curr_obj, prev_obj):
                    distance = self.calculate_distance(curr_obj['bbox'], prev_obj['bbox'])
                    speed = (distance * fps) / 100
                    speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0
    
    def is_same_vehicle(self, obj1, obj2, iou_threshold=0.3):
        if obj1['class_name'] != obj2['class_name']:
            return False
        
        iou = self.calculate_iou(obj1['bbox'], obj2['bbox'])
        return iou > iou_threshold
    
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
    
    def calculate_distance(self, bbox1, bbox2):
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance