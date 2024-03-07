import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

recognizer_model_path = 'models/gesture_recognizer_emptyNone.task'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']
# LABELS = ['🤙', '👎', '✊', '👍', '🤐', '👌', '☝', '🖐', '✌', '🤘', '✋', '🤚']

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
capture = cv2.VideoCapture(0)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        for gesture in result.gestures:
            for hand in gesture:
                category_name = hand.category_name
                score = hand.score
        print(f'{timestamp_ms}: {category_name}, {score}')


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=recognizer_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Webcam', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
capture.release()
cv2.destroyAllWindows()
