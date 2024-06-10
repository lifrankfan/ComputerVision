import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,
                 mode=False,
                 max_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.9,
                 min_tracking_confidence=0.9):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, 
                                         self.max_hands, 
                                         self.model_complexity,
                                         self.min_detection_confidence, 
                                         self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    # Find hands in the image
    def findHands(self, img, draw=True, flip=True):
        # cv2 uses BGR, mediapipe uses RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Iterate through hands and display them
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img
    
    # Find the position of hand landmarks
    def findPosition(self, img, hand_num=0, draw=True):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                # convert from decimal to pixel position
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        
        return landmarks

def main():
    prev_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findHands(img, flip=True)
        landmarks = detector.findPosition(img, draw=True)
        print(landmarks)

        # Calculate fps
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
