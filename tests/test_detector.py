import unittest
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from smartcity_vision.core.object_detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector(confidence_threshold=0.5)
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
    
    def test_object_detection(self):
        objects = self.detector.detect_objects(self.sample_image)
        self.assertIsInstance(objects, list)
    
    def test_filter_urban_objects(self):
        test_objects = [
            {'class_name': 'car', 'bbox': [10, 10, 50, 50], 'confidence': 0.8},
            {'class_name': 'person', 'bbox': [100, 100, 120, 140], 'confidence': 0.9},
            {'class_name': 'dog', 'bbox': [200, 200, 220, 240], 'confidence': 0.7}
        ]
        
        filtered = self.detector.filter_urban_objects(test_objects)
        self.assertEqual(len(filtered), 2)

if __name__ == '__main__':
    unittest.main()