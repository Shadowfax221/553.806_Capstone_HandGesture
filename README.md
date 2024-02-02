# 553.806_Capstone_HandGesture

timeline: https://docs.google.com/spreadsheets/d/1NNpIA2gp-X00Ras_H9GhWfCc83riU-bNeHhZ01tW40k/edit#gid=145052701

## Week 1
- [x] Dataset selection

dataset source: https://github.com/hukenovs/hagrid

## Week 2
- [x] label matching and selection
- [x] reduce dataset size
- [x] image resize and gray-scaling 

### label matching and selection

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

### reduce dataset size

The original dataset has 5000 to 30000+ pictures for each class. Reduce each to 100 for testing on resizing and gray-scaling. Reduced dataset is in \dataset. 
