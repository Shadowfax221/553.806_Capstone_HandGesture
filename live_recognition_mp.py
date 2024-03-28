import copy
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


    def draw_landmarks_on_image(self, annotated_image, hand_landmarks):
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
    
    def calc_bounding_rect(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]


    def draw_bounding_rect(self, annotated_image, detection_result):
        for hand_landmarks in detection_result.hand_landmarks:
            brect = self.calc_bounding_rect(annotated_image, hand_landmarks)
        cv2.rectangle(annotated_image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 1)
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
                annotated_image = copy.deepcopy(frame)
                
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, timestamp)

                if self.results is not None:
                    for hand_landmarks in self.results.hand_landmarks:
                        annotated_image = self.draw_landmarks_on_image(annotated_image, hand_landmarks)
                        # annotated_image = self.draw_bounding_rect(annotated_image, self.results)
                    cv2.imshow('Show', annotated_image)
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