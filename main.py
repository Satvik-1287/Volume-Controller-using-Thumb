import cv2
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

class HandGestureVolumeControl:
    def __init__(self):
        # Initialize MediaPipe Hands and Drawing Utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Get the audio interface
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.current_volume = self.volume.GetMasterVolumeLevelScalar()

    def detect_gesture(self, landmarks):
        # Extract landmarks
        landmarks = [landmark for landmark in landmarks.landmark]
        thumb, index = landmarks[4], landmarks[8]
        
        thumb_extended = thumb.y < landmarks[0].y
        index_extended = index.y < landmarks[0].y

        # Determine gesture
        if thumb_extended and index_extended:
            return "Increase Volume"
        elif not thumb_extended and not index_extended:
            return "Decrease Volume"
        else:
            return "Unknown"

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                gesture = self.detect_gesture(hand_landmarks)
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if gesture == "Increase Volume":
                    self.set_volume(min(self.current_volume + 0.1, 1.0))
                elif gesture == "Decrease Volume":
                    self.set_volume(max(self.current_volume - 0.1, 0.0))
        
        return frame

    def set_volume(self, volume_level):
        self.volume.SetMasterVolumeLevelScalar(volume_level, None)
        self.current_volume = volume_level

def main():
    cap = cv2.VideoCapture(0)
    volume_control = HandGestureVolumeControl()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = volume_control.process_frame(frame)
        cv2.imshow('Hand Gesture Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
