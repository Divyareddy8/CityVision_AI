import argparse
import cv2
import time
import json
from datetime import datetime

from smartcity_vision.core.object_detector import ObjectDetector
from smartcity_vision.core.traffic_analyzer import TrafficAnalyzer
from smartcity_vision.core.crowd_density import CrowdDensityAnalyzer
from smartcity_vision.core.parking_analyzer import ParkingAnalyzer
from smartcity_vision.core.pedestrian_tracker import PedestrianTracker
from smartcity_vision.utils.config_loader import ConfigLoader
from smartcity_vision.utils.video_processor import VideoProcessor
from smartcity_vision.utils.visualization import Visualization

def main():
    parser = argparse.ArgumentParser(description='SmartCity Vision System')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--headless', action='store_true', help='Run without display')
    
    args = parser.parse_args()
    
    config_loader = ConfigLoader(args.config)
    
    print("Initializing SmartCity Vision System...")
    
    detector = ObjectDetector(
        model_type=config_loader.get('object_detection.model_type'),
        confidence_threshold=config_loader.get('object_detection.confidence_threshold')
    )
    
    traffic_analyzer = TrafficAnalyzer(config_loader.get('traffic_analysis'))
    crowd_analyzer = CrowdDensityAnalyzer(config_loader.get('crowd_analysis.method'))
    parking_analyzer = ParkingAnalyzer()
    pedestrian_tracker = PedestrianTracker()
    visualizer = Visualization()
    
    video_processor = VideoProcessor(args.source)
    video_processor.start()
    
    if args.output:
        frame_size = video_processor.get_frame_size()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, 20.0, frame_size)
    
    print("Starting analysis...")
    
    try:
        while True:
            frame = video_processor.read()
            if frame is None:
                break
            
            objects = detector.detect_objects(frame)
            urban_objects = detector.filter_urban_objects(objects)
            
            traffic_analysis = traffic_analyzer.analyze_traffic_flow(urban_objects, frame.shape)
            crowd_analysis = crowd_analyzer.analyze_crowd_density(urban_objects, frame.shape)
            parking_analysis = parking_analyzer.analyze_parking_occupancy(urban_objects, frame)
            
            pedestrian_tracks = pedestrian_tracker.update_tracks(urban_objects)
            pedestrian_analysis = pedestrian_tracker.analyze_pedestrian_flow(pedestrian_tracks)
            
            annotated_frame = visualizer.draw_detections(frame, urban_objects)
            annotated_frame = visualizer.draw_traffic_analysis(annotated_frame, traffic_analysis)
            annotated_frame = visualizer.draw_crowd_density(annotated_frame, crowd_analysis)
            
            if args.output:
                out.write(annotated_frame)
            
            if not args.headless:
                cv2.imshow('SmartCity Vision', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            print(f"Traffic: {traffic_analysis['congestion_level']} | "
                  f"Crowd: {crowd_analysis['density_level']} | "
                  f"Parking: {parking_analysis['available_spots']} available")
    
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        video_processor.stop()
        if args.output:
            out.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()