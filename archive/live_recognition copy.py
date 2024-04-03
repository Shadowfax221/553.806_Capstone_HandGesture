import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

landmarker_model_path = 'models/hand_landmarker.task'
classifier_model_path='models/keypoint_classifier_part1.tflite'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
capture = cv2.VideoCapture(0)


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            landmark_list = [] 
            for landmark in hand_landmarks:
                landmark_list.extend([landmark.x, landmark.y, landmark.z]) 

        if len(landmark_list)==63:
            interpreter = tf.lite.Interpreter(model_path=classifier_model_path,
                                              num_threads=1)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_details_tensor_index = input_details[0]['index']
            interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            interpreter.invoke()
            output_details_tensor_index = output_details[0]['index']
            result = interpreter.get_tensor(output_details_tensor_index)
            result_index = np.argmax(np.squeeze(result))
            score = np.squeeze(result)[result_index]

            print(f'{timestamp_ms}: {LABELS[result_index]}, {score}')


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmarker_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Webcam', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
capture.release()
cv2.destroyAllWindows()