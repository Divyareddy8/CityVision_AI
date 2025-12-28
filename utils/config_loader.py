import yaml
import os

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'object_detection': {
                'model_type': 'yolov5',
                'confidence_threshold': 0.5,
                'target_classes': ['person', 'car', 'bus', 'truck', 'motorcycle']
            },
            'traffic_analysis': {
                'congestion_thresholds': {'low': 10, 'medium': 30, 'high': 50},
                'update_interval': 30
            },
            'crowd_analysis': {
                'method': 'counting',
                'density_thresholds': {'low': 5, 'medium': 20, 'high': 50}
            },
            'parking_analysis': {
                'detection_method': 'grid',
                'grid_size': [10, 10]
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default