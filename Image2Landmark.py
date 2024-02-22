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



