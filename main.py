import os
import time
import ultralytics
import cv2
import pandas as pd 
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker = Tracker()
count = 0

cap = cv2.VideoCapture('video.mp4')

down = {}
up = {}
counter_down = []
counter_up = []

object_info = {}

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    detections = results[0].boxes.data.detach().cpu().numpy()
    px = pd.DataFrame(detections).astype("float")

    list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        confidence = row[4]
        class_id = int(row[5])
        label = class_list[class_id]

        if 'car' in label:
            list.append([x1, y1, x2, y2, confidence, label])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id, confidence, label = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        # Store the latest position in object_info so bounding box follows the object
        object_info[id] = {
            'bbox': (x3, y3, x4, y4),
            'label': label,
            'confidence': confidence,
            'speed': object_info[id]['speed'] if id in object_info else 0
        }

        # Speed calculations when object crosses the lines
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()

        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    object_info[id]['speed'] = int(a_speed_kh)

        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()

        if id in up and red_line_y < (cy + offset) and red_line_y > (cy - offset):
            elapsed_time = time.time() - up[id]
            if id not in counter_up:
                counter_up.append(id)
                distance = 10
                speed_kmh = (distance / elapsed_time) * 3.6
                object_info[id]['speed'] = int(speed_kmh)

    for id, info in object_info.items():
        x3, y3, x4, y4 = info['bbox']
        speed = info['speed']
        label = info['label']
        confidence = info['confidence']

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {id}', (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x3, y3 - 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Speed: {speed} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw counter boxes and lines
    text_color = (0, 0, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, 'Red Line', (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
    cv2.putText(frame, 'Blue Line', (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f'Going Down - {len(counter_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f'Going Up - {len(counter_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save and display frame
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)
    out.write(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
