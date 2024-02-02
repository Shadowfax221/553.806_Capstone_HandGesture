import os
import shutil
import random

source_dir = "F:/下载/hagrid_dataset_512/hagrid_dataset_512"
target_dir = "C:/Users/renyi/OneDrive/JHU/Spring 2024/Capstone/553.806_Capstone_HandGesture/dataset"

os.makedirs(target_dir, exist_ok=True)

for subdir in os.listdir(source_dir):
    source_subdir = os.path.join(source_dir, subdir)
    
    if os.path.isdir(source_subdir):
        images = os.listdir(source_subdir)
        
        selected_images = random.sample(images, min(100, len(images)))
        
        target_subdir = os.path.join(target_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)

        for image in selected_images:
            source_image = os.path.join(source_subdir, image)
            target_image = os.path.join(target_subdir, image)
            shutil.copy(source_image, target_image)
