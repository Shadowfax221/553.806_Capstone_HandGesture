import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

recognizer_model_path = 'models/gesture_recognizer_emptyNone.task'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']
# LABELS = ['🤙', '👎', '✊', '👍', '🤐', '👌', '☝', '🖐', '✌', '🤘', '✋', '🤚']

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode




class Mediapipe_BodyModule():
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None


    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        #hand_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the pose landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks_proto,
                                                    self.mp_hands.HAND_CONNECTIONS,
                                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                        color=(255, 0, 255), thickness=4, circle_radius=2),
                                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                        color=(20, 180, 90), thickness=2, circle_radius=2)
            )
        return annotated_image
    
    # Create a gesture recognizer instance with the live stream mode:
    def print_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        # print('pose landmarker result: {}'.format(result))
        self.results = result
        #print(type(result))

        if result.hand_landmarks:
            for gesture in result.gestures:
                for hand in gesture:
                    category_name = hand.category_name
                    score = hand.score
            print(f'{timestamp_ms}: {category_name}, {score}')
    
    def main(self):
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=recognizer_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.print_result)
        
        capture = cv2.VideoCapture(0)

        timestamp = 0
        with GestureRecognizer.create_from_options(options) as recognizer:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    print("Ignoring empty frame")
                    break
                frame = cv2.flip(frame, 1)
                
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, timestamp)

                if self.results is not None:
                    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.results)
                    #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    cv2.imshow('Show', annotated_image)
                    # print("showing detected image")
                else:
                    cv2.imshow('Show', frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")
                    break
                        
            capture.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()