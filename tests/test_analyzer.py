import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from smartcity_vision.core.traffic_analyzer import TrafficAnalyzer
from smartcity_vision.core.crowd_density import CrowdDensityAnalyzer

class TestTrafficAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrafficAnalyzer()
        self.sample_objects = [
            {'class_name': 'car', 'bbox': [10, 10, 60, 40]},
            {'class_name': 'bus', 'bbox': [100, 50, 180, 80]},
            {'class_name': 'person', 'bbox': [200, 150, 220, 190]}
        ]
    
    def test_traffic_analysis(self):
        analysis = self.analyzer.analyze_traffic_flow(self.sample_objects, (480, 640))
        self.assertIn('vehicle_count', analysis)
        self.assertIn('congestion_level', analysis)

class TestCrowdAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CrowdDensityAnalyzer()
        self.sample_objects = [
            {'class_name': 'person', 'bbox': [10, 10, 30, 50]},
            {'class_name': 'person', 'bbox': [100, 100, 120, 140]},
            {'class_name': 'car', 'bbox': [200, 200, 250, 230]}
        ]
    
    def test_crowd_analysis(self):
        analysis = self.analyzer.analyze_crowd_density(self.sample_objects, (480, 640))
        self.assertIn('total_people', analysis)
        self.assertIn('density_level', analysis)

if __name__ == '__main__':
    unittest.main()