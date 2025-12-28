import cv2
import numpy as np
import os

def create_sample_video(output_path='data/sample_video.mp4', duration=10, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        num_cars = np.random.randint(0, 10)
        for _ in range(num_cars):
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 30)
            cv2.rectangle(frame, (x, y), (x + 50, y + 30), (0, 255, 0), -1)
        
        num_people = np.random.randint(0, 15)
        for _ in range(num_people):
            x = np.random.randint(0, width - 20)
            y = np.random.randint(0, height - 40)
            cv2.rectangle(frame, (x, y), (x + 20, y + 40), (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")