import os
import shutil
import random

# 设置原始目录和目标目录
source_dir = "F:/下载/hagrid_dataset_512/hagrid_dataset_512"
target_dir = "C:/Users/renyi/OneDrive/JHU/Spring 2024/Capstone/553.806_Capstone_HandGesture/dataset"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历原始目录中的每个子目录
for subdir in os.listdir(source_dir):
    source_subdir = os.path.join(source_dir, subdir)
    
    # 确保它是一个目录
    if os.path.isdir(source_subdir):
        images = os.listdir(source_subdir)
        
        # 选择100张图片（随机或前100张）
        selected_images = random.sample(images, min(100, len(images)))
        
        # 创建目标子目录
        target_subdir = os.path.join(target_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)

        # 复制图片到新目录
        for image in selected_images:
            source_image = os.path.join(source_subdir, image)
            target_image = os.path.join(target_subdir, image)
            shutil.copy(source_image, target_image)
