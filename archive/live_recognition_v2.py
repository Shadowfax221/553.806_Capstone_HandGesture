import argparse
import copy
import itertools
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf

landmarker_model_path = 'models/hand_landmarker.task'
CLASSIFIER_MODEL_PATH='models/keypoint_classifier.tflite'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()
    return args


def main():
    # Argument parsing #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


    #  ########################################################################
    while True:
        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image) 
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Hand sign classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                mp_drawing.draw_landmarks(debug_image, hand_landmarks,
                                                mp_hands.HAND_CONNECTIONS,
                                                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                    color=(255, 0, 255), thickness=4, circle_radius=2),
                                                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                    color=(20, 180, 90), thickness=2, circle_radius=2)
                                                )
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    LABELS[hand_sign_id],
                )
        
        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


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


def calc_landmark_list(image, landmarks):
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_list.append([landmark.x, landmark.y, landmark.z])
    return landmark_list


def apply_rotation(label, points):
    num_points = 21
    point_0 = points[0]
    point_8 = points[8]
    if label == 'dislike' or label == 'like':
        return points  # No rotation
    elif label == 'call':
        # Rotate to be parallel with X-axis
        line = point_8 - point_0
        angle = np.arctan2(line[2], line[0])  # Consider the X-Z plane
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    else:
        # Rotate to be parallel with Y-axis
        line = point_8 - point_0
        angle = np.arctan2(line[2], line[1])  # Consider the Y-Z plane
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    # Apply rotation
    for i in range(num_points):
        points[i] = np.dot(rotation_matrix, points[i])
    return points


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2] 
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def keypoint_classifier(landmark_list):
    interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_MODEL_PATH,
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
    return result_index


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image









if __name__ == '__main__':
    main()
