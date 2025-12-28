import cv2
import numpy as np
import threading
import queue
import time

class VideoProcessor:
    def __init__(self, source=0, buffer_size=64):
        self.source = source
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.cap = None
        
    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source)
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
    
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def get_frame_size(self):
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (640, 480)

class MultiCameraProcessor:
    def __init__(self, camera_sources):
        self.camera_sources = camera_sources
        self.processors = {}
        
        for cam_id, source in camera_sources.items():
            self.processors[cam_id] = VideoProcessor(source)
    
    def start_all(self):
        for processor in self.processors.values():
            processor.start()
    
    def stop_all(self):
        for processor in self.processors.values():
            processor.stop()
    
    def read_all(self):
        frames = {}
        for cam_id, processor in self.processors.items():
            frame = processor.read()
            if frame is not None:
                frames[cam_id] = frame
        return frames