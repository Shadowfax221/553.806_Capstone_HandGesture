import os
import random
import json
import shutil
from PIL import Image

NUM_EXAMPLES = 125
DATASET_NAME = 'dataset_selected125'
source_dir = "E:/MyDatasets/hagrid_dataset_512"
target_dir = f"C:/Users/Ian/git/553.806_Capstone_HandGesture/datasets/{DATASET_NAME}"
annotations_dir = "C:/Users/Ian/git/553.806_Capstone_HandGesture/annotations/train"
labels = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']     # 12 gestures: 🤙, 👎, ✊, 👍, 🤐, 👌, ☝, 🖐, ✌, 🤘, ✋, 🤚

# Remove the directory and all its contents
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.makedirs(target_dir, exist_ok=True)

# Function to read JSON annotations
def read_annotations(label):
    annotation_path = os.path.join(annotations_dir, f"{label}.json")
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)
    return annotations

# Function to crop image based on the bounding box
def crop_image(image_path, bbox):
    with Image.open(image_path) as img:
        width, height = img.size
        x0 = int(bbox[0] * width)
        y0 = int(bbox[1] * height)
        x1 = x0 + int(bbox[2] * width)
        y1 = y0 + int(bbox[3] * height)
        return img.crop((x0, y0, x1, y1))

# Function to save and crop selected images
def save_image(label_name, images, no_gesture=False, target_dir=target_dir):
    if no_gesture:
        target_subdir = os.path.join(target_dir, "none")
    else:
        target_subdir = os.path.join(target_dir, label_name)
    os.makedirs(target_subdir, exist_ok=True)
    for image_name, bbox in images:
        source_image_path = os.path.join(source_dir, label_name, image_name)
        cropped_image = crop_image(source_image_path, bbox)
        target_image_path = os.path.join(target_subdir, image_name)
        cropped_image.save(target_image_path)


# Iterate over labels (annotation files)
for label_name in labels:
    annotations = read_annotations(label_name)

    # selected_no_gesture = 0
    selected_images = []
    # no_gesture_images = []
    keys = list(annotations.keys())
    selected_keys = random.sample(keys, min(NUM_EXAMPLES, len(keys)))
    for key in selected_keys:
        image_name = f"{key}.jpg"
        image_path = os.path.join(source_dir, label_name, image_name)
        if os.path.exists(image_path):
            key_labels = annotations[key]['labels']
            key_bboxes = annotations[key]['bboxes']
            selected_images.append((image_name, key_bboxes[key_labels.index(label_name)]))
            # if "no_gesture" in key_labels:
            #     no_gesture_images.append((image_name, key_bboxes[key_labels.index("no_gesture")]))
            #     selected_no_gesture += 1

    print(label_name, len(selected_images))

    # Copy and crop selected images and no_gesture images
    save_image(label_name, selected_images)
    # save_image(label_name, no_gesture_images, no_gesture=True)


print('DONE')