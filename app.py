import cv2
import torch
import numpy as np
from flask import Flask, jsonify, Response
import threading
import time
from flask_cors import CORS


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

parking_spaces = [
    [200, 300, 300, 400],
    [320, 300, 420, 400],
    [440, 300, 540, 400],
    [560, 300, 660, 400],
    [680, 300, 780, 400],
    [800, 300, 900, 400],
]

latest_free_slots = 0
latest_slot_status = [True] * len(parking_spaces)
latest_frame = None  

def process_video():
    global latest_free_slots, latest_slot_status, latest_frame
    
    video_path = 'vid.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        results = model(frame)
        
        parking_status = [True] * len(parking_spaces)
        
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 2:  
                x1, y1, x2, y2 = map(int, box)
                for i, space in enumerate(parking_spaces):
                    if (x1 < space[2] and x2 > space[0] and y1 < space[3] and y2 > space[1]):
                        parking_status[i] = False
        
        latest_free_slots = sum(parking_status)
        latest_slot_status = parking_status
        
        for i, space in enumerate(parking_spaces):
            color = (0, 255, 0) if parking_status[i] else (0, 0, 255)
            cv2.rectangle(frame, (space[0], space[1]), (space[2], space[3]), color, 2)
        
        latest_frame = frame
        
        time.sleep(0.1)

app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return "Parking Status API. Use /parking_status to get the status of parking spaces."

@app.route('/parking_status', methods=['GET'])
def get_parking_status():
    locations = [
        {
            'lat': 12.9132462,
            'lng': 77.5635128,
            'name': 'New York City',
            'description': 'The big apple.',
            'free': latest_free_slots,
            'floors': {
                '1': latest_slot_status
            },
        },
        {
            'lat': 12.9094733,
            'lng': 77.5644549,
            'name': 'Los Angeles',
            'description': 'The city of angels.',
            'free': latest_free_slots,  
            'floors': {
                '1': latest_slot_status
            },
        },
    ]
    return jsonify(locations)


@app.route('/parking_frame', methods=['GET'])
def get_parking_frame():
    global latest_frame
    if latest_frame is None:
        return "No frame available", 503
    
    ret, jpeg = cv2.imencode('.jpg', latest_frame)
    if not ret:
        return "Failed to encode frame", 500
    
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    app.run(host='0.0.0.0', port=8080)