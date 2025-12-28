import cv2
import numpy as np
from collections import defaultdict, deque

class PedestrianTracker:
    def __init__(self, max_age=30):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
        self.trajectories = defaultdict(lambda: deque(maxlen=50))
        
    def update_tracks(self, detections):
        current_tracks = {}
        
        for detection in detections:
            if detection['class_name'] == 'person':
                track_id = self.assign_track_id(detection)
                bbox = detection['bbox']
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                
                self.trajectories[track_id].append(center)
                
                current_tracks[track_id] = {
                    'bbox': bbox,
                    'center': center,
                    'age': 0,
                    'trajectory': list(self.trajectories[track_id])
                }
        
        self.cleanup_old_tracks(current_tracks)
        return current_tracks
    
    def assign_track_id(self, detection):
        best_match = None
        best_iou = 0.3
        
        for track_id, track in self.tracks.items():
            iou = self.calculate_iou(track['bbox'], detection['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = track_id
        
        if best_match is not None:
            return best_match
        else:
            new_id = self.next_id
            self.next_id += 1
            return new_id
    
    def cleanup_old_tracks(self, current_tracks):
        tracks_to_remove = []
        
        for track_id in self.tracks:
            if track_id not in current_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
            else:
                self.tracks[track_id] = current_tracks[track_id]
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.trajectories:
                del self.trajectories[track_id]
    
    def analyze_pedestrian_flow(self, tracks):
        total_pedestrians = len(tracks)
        movement_patterns = {}
        
        for track_id, track in tracks.items():
            if len(track['trajectory']) >= 2:
                start_point = track['trajectory'][0]
                end_point = track['trajectory'][-1]
                
                direction = self.calculate_direction(start_point, end_point)
                speed = self.calculate_speed(track['trajectory'])
                
                movement_patterns[track_id] = {
                    'direction': direction,
                    'speed': speed,
                    'distance_traveled': self.calculate_distance_traveled(track['trajectory'])
                }
        
        return {
            'total_pedestrians': total_pedestrians,
            'movement_patterns': movement_patterns,
            'average_speed': np.mean([p['speed'] for p in movement_patterns.values()]) if movement_patterns else 0
        }
    
    def calculate_direction(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if angle < 0:
            angle += 360
        
        if 45 <= angle < 135:
            return 'Down'
        elif 135 <= angle < 225:
            return 'Left'
        elif 225 <= angle < 315:
            return 'Up'
        else:
            return 'Right'
    
    def calculate_speed(self, trajectory):
        if len(trajectory) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            distance = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            total_distance += distance
        
        return total_distance / len(trajectory)
    
    def calculate_distance_traveled(self, trajectory):
        total_distance = 0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            distance = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            total_distance += distance
        
        return total_distance