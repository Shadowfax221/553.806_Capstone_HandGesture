import os
import shutil
import random
import json
from PIL import Image

source_dir = "E:/MyDatasets/hagrid_dataset_512"
target_dir = "C:/Users/Ian/git/553.806_Capstone_HandGesture/dataset_boxed"
annotations_dir = "C:/Users/Ian/git/553.806_Capstone_HandGesture/annotations/train"

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

# Iterate over labels (annotation files)
for label in os.listdir(annotations_dir):
    label_name = label.split('.')[0]  # Remove file extension to get the label
    annotations = read_annotations(label_name)

    selected_images = []
    keys = list(annotations.keys())
    selected_keys = random.sample(keys, min(100, len(keys)))
    for key in selected_keys:
        image_name_with_ext = f"{key}.jpg"
        image_path = os.path.join(source_dir, label_name, image_name_with_ext)
        if os.path.exists(image_path):
            label_idx = annotations[key]['labels'].index(label_name)
            selected_images.append((image_name_with_ext, annotations[key]['bboxes'][label_idx]))
    print(label_name, len(selected_images))

    # Copy and crop selected images
    target_subdir = os.path.join(target_dir, label_name)
    os.makedirs(target_subdir, exist_ok=True)
    for image_name, bbox in selected_images:
        source_image_path = os.path.join(source_dir, label_name, image_name)
        cropped_image = crop_image(source_image_path, bbox)
        target_image_path = os.path.join(target_subdir, image_name)
        cropped_image.save(target_image_path)
print('Done')