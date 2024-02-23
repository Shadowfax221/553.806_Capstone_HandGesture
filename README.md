# 553.806_Capstone_HandGesture

timeline: https://docs.google.com/spreadsheets/d/1NNpIA2gp-X00Ras_H9GhWfCc83riU-bNeHhZ01tW40k/edit#gid=145052701

## Week 1
- [x] Dataset selection

dataset source: https://github.com/hukenovs/hagrid

## Week 2
- [x] label matching and selection
- [x] reduce dataset size
- [x] image resize and gray-scaling 

### Label matching and selection

18 classes of gesture of gestures in dataset: 

![gestures](https://github.com/hukenovs/hagrid/raw/master/images/gestures.jpg)

mapping each gesture to emoji: 
* ref to https://unicode.org/emoji/charts/full-emoji-list.html

| dataset     | emoji         | shortcode                 | Unicode     | comment |
|-------------|---------------|---------------------------|-------------|---------|
| call        | 🤙            | `:call_me_hand:`          | U+1F919     |         |
| dislike     | 👎            | `:thumbs_down:`           | U+1F44E     |         |
| fist        | ✊            | `:raised_fist:`           | U+270A      |         |
| ~~four~~    | 🖖            | `:vulcan_salute:`         | U+1F596     | weak emoji        |
| like        | 👍            | `:thumbs_up:`             | U+1F44D     |         |
| mute        | 🤐            | `:zipper_mouth_face:`     | U+1F910     | uncommon |
| ok          | 👌            | `:ok_hand:`               | U+1F44C     |         |
| one         | ☝             | `:index_pointing_up:`     | U+261D      |         |
| palm        | 🖐            | `:raised_hand_with_fingers_splayed:` | U+1F590 |         |
| peace       | ✌             | `:victory_hand:`          | U+270C      |         |
| ~~peace inv.~~  | 🤘       | `:sign_of_the_horns:`     | U+1F918     | weak emoji     |
| rock        | 🤘            | `:sign_of_the_horns:`     | U+1F918     |         |
| stop        | ✋            | `:raised_hand:`           | U+270B      | similar to stop inv        |
| stop inv.   | 🤚            | `:raised_back_of_hand:`   | U+1F91A     | similar to stop        |
| ~~three~~   | 🤟            | `:love-you_gesture:`      | U+1F91F     | weak emoji        |
| ~~three 2~~ |             |      |     | no such emoji        |
| ~~two up~~  | ✌             | `:victory_hand:`          | U+270C      | weak emoji        |
| ~~two up inv.~~ |         |      |      | no such emoji         |

### Reducing dataset size

The original dataset has 5000 to 30000+ pictures for each class. Reduce each to 100 for testing on resizing and gray-scaling. Reduced dataset is in \dataset. 

### Gray-scaling and resizing

The hand gesture dataset was processed through a two-step procedure: 
1. Each image was converted from its original RGB format to grayscale, effectively reducing the color information to shades of gray, which simplifies the data while retaining essential features.
2. All images were resized to a uniform dimension, ensuring consistency across the dataset, which is crucial for effective training and analysis in machine learning models.

These procedures enhances computational efficiency and standardizes the input for subsequent tasks.


## Week 3

- [x] Background removal
- [x] Skeleton structures detection

### Background removal

Reference and based on the work: https://github.com/Gogul09/gesture-recognition?tab=readme-ov-file

This code is designed to detect and segment a hand in real-time using a webcam feed with OpenCV in Python. 

* Original script:

The original script performs real-time hand detection using a webcam. It initializes the webcam and defines a Region of Interest (ROI) for detecting the hand. Each frame captured from the webcam is processed to isolate this ROI, which is then converted to grayscale and blurred for noise reduction. The script uses a running average method to model the background and segments the hand by identifying differences between the background and current frame. When a hand is detected within the ROI, it's outlined with contours, and the result is displayed in real-time. The script allows continuous hand detection until the user exits by pressing a designated key.

* Functionalities added:

The enhancements to the script introduce the capability to capture and save snapshots of the hand detection process. By pressing a specific key, the user can take a snapshot of both the ROI and the processed image showing the detected hand. These images are saved in a newly created "snapshots" folder in the script's directory. Each saved image is uniquely named using a counter to avoid overwriting. This feature enables users to save specific moments of the hand detection process for further analysis or record-keeping.

![image](https://github.com/Shadowfax221/553.806_Capstone_HandGesture/assets/126203843/d2e85340-3031-42f5-939e-f02f1d5fe278)


## Week 4
- [x] Train model using own dataset (15 gestures, 100 images for each)
- [x] Valid model and display result

![image](https://github.com/Shadowfax221/553.806_Capstone_HandGesture/assets/48790906/ac8afc26-05dc-469c-b553-91496b21ff4d)

TODO next week: 
- [ ] Data cleaning: larger box size; remove low resolution image
- [ ] Increase training size
- [ ] Play with parameters


## Week 5

- [x] New Data Selection
- [x] Hyperparameter tuning on mediapipe model
- [x] Livestream testing
- [x] Self-implemented neural network with tensorflow

### New Data Selection

Expanded our dataset to include 1,000 images per category, consolidating them into 11 distinct labels/categories.

labels: 
| idx | dataset     | emoji         | shortcode                 | Unicode     | comment |
|-----|-------------|---------------|---------------------------|-------------|---------|
|0| call        | 🤙            | `:call_me_hand:`          | U+1F919     |         |
|1| dislike     | 👎            | `:thumbs_down:`           | U+1F44E     |         |
|2| fist        | ✊            | `:raised_fist:`           | U+270A      |         |
|3| like        | 👍            | `:thumbs_up:`             | U+1F44D     |         |
|4| mute        | 🤐            | `:zipper_mouth_face:`     | U+1F910     | uncommon |
|5| ok          | 👌            | `:ok_hand:`               | U+1F44C     |         |
|6| one         | ☝             | `:index_pointing_up:`     | U+261D      |         |
|7| palm        | 🖐            | `:raised_hand_with_fingers_splayed:` | U+1F590 |         |
|8| peace       | ✌             | `:victory_hand:`          | U+270C      |         |
|9| rock        | 🤘            | `:sign_of_the_horns:`     | U+1F918     |         |
|10| stop        | ✋            | `:raised_hand:`           | U+270B      | similar to stop inv        |
|11| stop inv.   | 🤚            | `:raised_back_of_hand:`   | U+1F91A     | similar to stop        |


### Hyperparamter tuning on mediapipe model

Conducted hyperparameter tuning that encompasses adjustments in:

* Network shape
* Dropout rate
* Batch size

### Livestream Testing

Evaluated our custom-trained model using a live stream feed and output. This was facilitated through [MediaPipe Studio](https://mediapipe-studio.webapps.google.com/studio/demo/gesture_recognizer), where we uploaded our custom model for testing.

Area for Enhancement: 
- The "none" label problem.

### Self-implemented neural network with tensorflow

Used tensorflow to build another model with tensorflow. Reference: [kivini](https://github.com/kinivi/hand-gesture-recognition-mediapipe). The process includes:
- Turning landmarks into csv files.
- Using data in csv file for training in custom tensorflow neural network.
- Testing result in a local livestream.

Area for Enhancement: 
- Low accuracy (need to optimize the model).
- Hand not detected in some cases.




  
