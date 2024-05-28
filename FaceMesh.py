import cv2
import mediapipe as mp
import time

class FaceMeshDetector:

    def __init__(self, mode=False, max_faces=2, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.max_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.drawSpec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, faceLms, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        self.drawSpec, self.drawSpec
                    )
                face = []
                for lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture("videos/6.mp4")
    prev_time = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
