import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



DATASET_NAME = 'dataset_selected'
DATASET_DIR = f"C:/Users/Ian/git/553.806_Capstone_HandGesture/datasets/{DATASET_NAME}"
labels = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']     # 12 gestures: 🤙, 👎, ✊, 👍, 🤐, 👌, ☝, 🖐, ✌, 🤘, ✋, 🤚
label_dict = {label: i for i, label in enumerate(labels)}

# Hand landmark detection setup
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# CSV file setup
csv_filename = 'keypoint.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    for label in labels:
        label_dir = os.path.join(DATASET_DIR, label)
        num_valid_image = 0
        for image_filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_filename)
            
            image = mp.Image.create_from_file(image_path)
            detection_result = detector.detect(image)

            # Prepare the CSV row
            row = [label_dict[label]]
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y])
            
            if len(row) == 43:
                csvwriter.writerow(row)
                num_valid_image += 1

        print(label_dir, num_valid_image, '/', 1000)
            

print("CSV file has been created:", csv_filename)