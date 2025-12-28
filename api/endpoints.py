from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import base64

router = APIRouter()

@router.post("/analyze/traffic")
async def analyze_traffic(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    from smartcity_vision.core.object_detector import ObjectDetector
    from smartcity_vision.core.traffic_analyzer import TrafficAnalyzer
    
    detector = ObjectDetector()
    traffic_analyzer = TrafficAnalyzer()
    
    objects = detector.detect_objects(image)
    analysis = traffic_analyzer.analyze_traffic_flow(objects, image.shape)
    
    return analysis

@router.post("/analyze/crowd")
async def analyze_crowd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    from smartcity_vision.core.object_detector import ObjectDetector
    from smartcity_vision.core.crowd_density import CrowdDensityAnalyzer
    
    detector = ObjectDetector()
    crowd_analyzer = CrowdDensityAnalyzer()
    
    objects = detector.detect_objects(image)
    analysis = crowd_analyzer.analyze_crowd_density(objects, image.shape)
    
    return analysis

@router.post("/analyze/parking")
async def analyze_parking(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    from smartcity_vision.core.object_detector import ObjectDetector
    from smartcity_vision.core.parking_analyzer import ParkingAnalyzer
    
    detector = ObjectDetector()
    parking_analyzer = ParkingAnalyzer()
    
    objects = detector.detect_objects(image)
    parking_spots = parking_analyzer.detect_parking_spots(image)
    analysis = parking_analyzer.analyze_parking_occupancy(objects, image)
    
    return analysis

# api/websocket_handler.py
from fastapi import WebSocket
import json
import asyncio

class WebSocketHandler:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.active_connections.remove(connection)