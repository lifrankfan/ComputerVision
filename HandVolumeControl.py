import cv2
import mediapipe as mp
import numpy as np
import time
import HandDetection as hd
import math
import alsaaudio

# audio control
mixer = alsaaudio.Mixer()
vol = 0

cam_width, cam_height = 640, 480
prev_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = hd.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarks = detector.findPosition(img)

    if len(landmarks) != 0:
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        y3 = landmarks[12][2]
        y4 = landmarks[9][2]
        # center of two x1, y1 and x2, y2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        print(y3, y4)
        
        # draw shapes
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
        length = math.hypot(x2 - x1, y2 - y1)
        
        if y3 > y4:        
            # Hand range: 200 - 15 range
            vol = np.interp(length, [20, 180], [0, 50])
            mixer.setvolume(int(vol))
    
    vol_bar = np.interp(vol, [0, 100], [400, 150])
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # fps
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break