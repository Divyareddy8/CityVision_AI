from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import json
import threading
import time

app = Flask(__name__)

class Dashboard:
    def __init__(self):
        self.traffic_data = []
        self.crowd_data = []
        self.parking_data = []
        self.latest_frame = None
        
    def generate_frames(self):
        from smartcity_vision.core.object_detector import ObjectDetector
        from smartcity_vision.core.traffic_analyzer import TrafficAnalyzer
        from smartcity_vision.utils.visualization import Visualization
        
        detector = ObjectDetector()
        traffic_analyzer = TrafficAnalyzer()
        visualizer = Visualization()
        
        camera = cv2.VideoCapture(0)
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            objects = detector.detect_objects(frame)
            traffic_analysis = traffic_analyzer.analyze_traffic_flow(objects, frame.shape)
            
            annotated_frame = visualizer.draw_detections(frame, objects)
            annotated_frame = visualizer.draw_traffic_analysis(annotated_frame, traffic_analysis)
            
            self.traffic_data.append(traffic_analysis)
            self.latest_frame = annotated_frame
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        camera.release()

dashboard = Dashboard()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(dashboard.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/traffic_data')
def get_traffic_data():
    return jsonify(dashboard.traffic_data[-50:])

@app.route('/api/crowd_data')
def get_crowd_data():
    return jsonify(dashboard.crowd_data[-50:])

@app.route('/api/parking_data')
def get_parking_data():
    return jsonify(dashboard.parking_data[-50:])

def run_dashboard():
    app.run(host='0.0.0.0', port=5000, debug=True)