
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

class CrowdDensityAnalyzer:
    def __init__(self, method='density_map'):
        self.method = method
        self.density_model = self.load_density_model() if method == 'density_map' else None
        
    def load_density_model(self):
        class DensityNet(nn.Module):
            def __init__(self):
                super(DensityNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
                self.density = nn.Conv2d(128, 1, 1)
                
            def forward(self, x):
                x = self.features(x)
                x = self.density(x)
                return x
        
        model = DensityNet()
        return model
    
    def analyze_crowd_density(self, objects, frame_shape):
        people = [obj for obj in objects if obj['class_name'] == 'person']
        
        if self.method == 'counting':
            return self.counting_based_density(people, frame_shape)
        elif self.method == 'clustering':
            return self.clustering_based_density(people, frame_shape)
        else:
            return self.density_map_estimation(people, frame_shape)
    
    def counting_based_density(self, people, frame_shape):
        total_people = len(people)
        frame_area = frame_shape[0] * frame_shape[1]
        
        density = (total_people / frame_area) * 100000
        
        if density < 5:
            density_level = 'Very Low'
        elif density < 20:
            density_level = 'Low'
        elif density < 50:
            density_level = 'Medium'
        elif density < 100:
            density_level = 'High'
        else:
            density_level = 'Very High'
        
        return {
            'total_people': total_people,
            'density_value': density,
            'density_level': density_level,
            'people_locations': [obj['bbox'] for obj in people]
        }
    
    def clustering_based_density(self, people, frame_shape):
        if not people:
            return {'total_people': 0, 'clusters': 0, 'avg_cluster_size': 0}
        
        centers = []
        for person in people:
            bbox = person['bbox']
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            centers.append(center)
        
        centers = np.array(centers)
        
        clustering = DBSCAN(eps=50, min_samples=2).fit(centers)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        
        return {
            'total_people': len(people),
            'clusters': n_clusters,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'cluster_distribution': cluster_sizes
        }
    
    def density_map_estimation(self, people, frame_shape):
        density_map = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for person in people:
            bbox = person['bbox']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            cv2.circle(density_map, (center_x, center_y), 30, 1, -1)
        
        total_density = np.sum(density_map)
        
        return {
            'density_map': density_map,
            'total_density': total_density,
            'hotspots': self.find_hotspots(density_map)
        }
    
    def find_hotspots(self, density_map, threshold=0.5):
        _, binary_map = cv2.threshold(density_map, threshold, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hotspots = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                hotspots.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': w * h,
                    'density': np.sum(density_map[y:y+h, x:x+w])
                })
        
        return hotspots
