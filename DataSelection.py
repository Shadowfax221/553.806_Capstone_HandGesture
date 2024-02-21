import os
import random
import json
import shutil
from PIL import Image

source_dir = "E:/MyDatasets/hagrid_dataset_512"
target_dir = "C:/Users/Ian/git/553.806_Capstone_HandGesture/dataset_selected"
annotations_dir = "C:/Users/Ian/git/553.806_Capstone_HandGesture/annotations/train"
labels = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']     # 12 gestures: 🤙, 👎, ✊, 👍, 🤐, 👌, ☝, 🖐, ✌, 🤘
NUM_EXAMPLES = 1000
NUM_NO_GESTURE = NUM_EXAMPLES // len(labels)

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


no_gesture_images = []

# Iterate over labels (annotation files)
for label_name in labels:
    annotations = read_annotations(label_name)

    selected_no_gesture = 0
    selected_images = []
    keys = list(annotations.keys())
    selected_keys = random.sample(keys, min(NUM_EXAMPLES, len(keys)))
    for key in selected_keys:
        image_name_with_ext = f"{key}.jpg"
        image_path = os.path.join(source_dir, label_name, image_name_with_ext)
        if os.path.exists(image_path):
            label_idx = annotations[key]['labels'].index(label_name)
            selected_images.append((image_name_with_ext, annotations[key]['bboxes'][label_idx]))
        if "no_gesture" in annotations[key]['labels'] and selected_no_gesture < NUM_NO_GESTURE:
            no_gesture_idx = annotations[key]['labels'].index("no_gesture")
            no_gesture_images.append((key, annotations[key]['bboxes'][no_gesture_idx]))
            selected_no_gesture += 1
    print(label_name, len(selected_images), selected_no_gesture)

    # Copy and crop selected images
    target_subdir = os.path.join(target_dir, label_name)
    os.makedirs(target_subdir, exist_ok=True)
    for image_name, bbox in selected_images:
        source_image_path = os.path.join(source_dir, label_name, image_name)
        cropped_image = crop_image(source_image_path, bbox)
        target_image_path = os.path.join(target_subdir, image_name)
        cropped_image.save(target_image_path)

# Process 'no_gesture' images
target_subdir = os.path.join(target_dir, "none")
os.makedirs(target_subdir, exist_ok=True)
for key, bbox in no_gesture_images:
    image_name_with_ext = f"{key}.jpg"
    source_image_path = os.path.join(source_dir, image_name_with_ext)
    if os.path.exists(source_image_path):
        cropped_image = crop_image(source_image_path, bbox)
        target_image_path = os.path.join(target_subdir, image_name_with_ext)
        cropped_image.save(target_image_path)

print('DONE')