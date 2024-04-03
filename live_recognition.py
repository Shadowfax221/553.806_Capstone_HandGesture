import copy
import itertools

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

landmarker_model_path = 'models/hand_landmarker.task'
classifier_model_path='models/keypoint_classifier.tflite'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class Mediapipe_HandModule():
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None


    def pre_process_landmark(self, landmarks):
        landmark_list = []
        # Convert to relative coordinates
        base_x, base_y, base_z = 0, 0, 0
        for idx, landmark in enumerate(landmarks.landmark):
            if idx == 0:
                base_x, base_y, base_z = landmark.x, landmark.y, landmark.z
            landmark_list.append([landmark.x-base_x, landmark.y-base_y, landmark.z-base_z])
        # Convert to a one-dimensional list
        landmark_list = list(itertools.chain.from_iterable(landmark_list))
        # Normalization
        landmark_list = landmark_list / np.max(np.abs(landmark_list))
        return landmark_list


    def draw_landmarks_on_image(self, annotated_image, hand_landmarks):
        self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(255, 0, 255), thickness=4, circle_radius=2),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(20, 180, 90), thickness=2, circle_radius=2)
        )
        return annotated_image
    

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def draw_bounding_rect(self, annotated_image, brect):
        cv2.rectangle(annotated_image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 1)
        return annotated_image
    

    def draw_info_text(self, image, brect, score, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)
        info_text = f"{score:.2f}"
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return image



    # Create a gesture landmarker instance with the live stream mode:
    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # print('pose landmarker result: {}'.format(result))
        self.results = result
    
    def main(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=landmarker_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.print_result)
        
        capture = cv2.VideoCapture(0)

        timestamp = 0
        with HandLandmarker.create_from_options(options) as landmarker:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    print("Ignoring empty frame")
                    break
                frame = cv2.flip(frame, 1)
                annotated_image = copy.deepcopy(frame)
                
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, timestamp)

                if self.results is not None:
                    for hand_landmarks in self.results.hand_landmarks:
                        # hand_landmarks set up ########################################################
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                        ])
                        # Hand Classification ##########################################################
                        pre_processed_landmarks = self.pre_process_landmark(hand_landmarks_proto)
                        self.pre_process_landmark_v2(hand_landmarks_proto)
                        # drawing part
                        annotated_image = self.draw_landmarks_on_image(annotated_image, hand_landmarks_proto)
                        brect = self.calc_bounding_rect(annotated_image, hand_landmarks_proto)
                        annotated_image = self.draw_bounding_rect(annotated_image, brect)
                        annotated_image = self.draw_info_text(
                            annotated_image,
                            brect,
                            0.9999,
                            "yes",
                        )
                        # print(self.results.gestures)
                        
                    cv2.imshow('Show', annotated_image)
                else:
                    cv2.imshow('Show', frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")
                    break
                        
            capture.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    body_module = Mediapipe_HandModule()
    body_module.main()