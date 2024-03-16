# UNO Card Detection

![UNO Card Detection Result](screenshot.png)

This repo is for AIROST Workshop.

UNO Dataset Description
```python
{
    "0":0,
    "1":1,
    "2":2,
    "3":3,
    "4":4,
    "5":5,
    "6":6,
    "7":7,
    "8":8,
    "9":9,
    "+4":0,
    "+2":11,
    "reverse":12,
    "skip":13,
    "wild":14
}
```

## How to clone this program

1. Open your command prompt
2. run `cd /d D:`
3. run `git clone https://github.com/shaunliew/uno-card-detection.git`
4. Open `uno-card-detection` folder using VSCode.
5. Switch branch to `workshop` branch
6. Open terminal in VSCode
7. create virtual env `python3.10 -m venv venv`
8. Activate virtual env `d:/uno-card-detection/venv/Scripts/Activate.ps1`
9. run `pip install -r requirements.txt`

Now you are good to go for this workshop. See you in the workshop.

## If you wanna download your own pip library

```
pip install opencv-python
pip install ultralytics
```

## How to run this program

```python
python main.py
```

## How the code works

### Step 1: Import the necessary libraries

```python
import cv2
from ultralytics import YOLO
```

We'll be using the OpenCV library (cv2) for video capture and image processing, and the Ultralytics library (YOLO) for object detection using YOLOv8.

### Step 2: Load the YOLOv8 model

```python
model = YOLO('uno-obb.pt')
```
Load the pre-trained YOLOv8 model ,`uno-obb.pt` for UNO card detection.

### Step 3: Open the webcam

```python
video_path = 0
cap = cv2.VideoCapture(video_path)
```

Open the video file using `OpenCV's VideoCapture` function. Set video_path to `0` if you want to use your webcam, or provide the path to a video file.

### Step 4: Define the UNO card classes and player regions

```python
UNO = {
    0: 0, 1: 1, 2: "+4", 3: "+2", 4: "reverse", 5: "skip",
    6: "wild", 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9
}

player_regions = {
    "Player 1": (0, 0, 640, 720),
    "Player 2": (640, 0, 1280, 720)
}
```

Define a dictionary `UNO` that maps the class indices to their corresponding card values. This `UNO` dictionary is based on the `class_id` and `class_name` from the public dataset. Also, define the regions for each player in the `player_regions` dictionary. For our case, we separate the player into left and right side.

### Step 5: Initialize player hands and latest card values

```python
player_hands = {
    "Player 1": [],
    "Player 2": []
}

latest_card_values = {
    "Player 1": "",
    "Player 2": ""
}
```

Initialize empty lists to store each player's hand and empty strings to store the latest card values for each player.

### Step 6: Code out the main logic

In this step, we'll dive into the main logic of our UNO card detection program. We'll process the video frames, detect the UNO cards using YOLOv8, and perform various operations based on the detected cards.

#### 6.1: Start the video processing loop

```python
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Video processing logic goes here
    else:
        break
```

We begin by starting a loop that continues as long as the video capture object (`cap`) is opened. Inside the loop, we read each frame from the video using `cap.read()`. If the frame is successfully read (`success` is `True`), we proceed with the video processing logic. If there are no more frames to read, we break out of the loop.

#### 6.2: Run YOLOv8 inference on the frame

```python
results = model(frame)[0]
```

We pass the current frame to our YOLOv8 model using model(frame). The model performs object detection on the frame and returns the detection results. We access the first element of the results ([0]) since we're processing a single frame.

#### 6.3: Iterate over the detected objects

```python
for result in results:
    if result is not None:
        # Extract the detected class and bounding box
        obb = result.obb
        detected_classes = obb.cls

        # Iterate over the detected classes
        for cls in detected_classes:
            # Object processing logic goes here
```

We iterate over each detected object (`result`) in the `results`. If the `result` is not None, we extract the detected class and bounding box information from `result.obb`. The `obb` (oriented bounding box) contains the coordinates and class information of the detected object. We access the detected classes using `obb.cls`.

#### 6.4: Process each detected UNO card

```python
card = UNO[int(cls)]

# Get the bounding box coordinates
bbox = obb.xyxy[0].cpu().numpy()
x1, y1, x2, y2 = bbox

# Determine the player based on the card location
player = None
for player_name, region in player_regions.items():
    if region[0] <= x1 <= region[2] and region[1] <= y1 <= region[3]:
        player = player_name
        break

if player is not None:
    # Add the card to the player's hand
    player_hands[player].append(card)

    # Update the latest card value for the player
    latest_card_values[player] = str(card)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Put the card name text above the bounding box
    cv2.putText(frame, str(card), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
```

For each detected class (`cls`), we perform the following steps:

- Map the class index to the corresponding UNO card value using the `UNO` dictionary.
- Extract the bounding box coordinates (`bbox`) of the detected card. We access the coordinates using `obb.xyxy[0]` and convert them to a NumPy array using `cpu().numpy()`.
- Determine the player who played the card based on the card's location. We iterate over the `player_regions dictionary` and check if the card's coordinates fall within any of the defined regions. If a match is found, we assign the corresponding player name to the `player`variable.
- If a player is identified, we add the detected card to the player's hand using `player_hands[player].append(card)` and update the latest card value for that player in the `latest_card_values` dictionary.

- We draw the bounding box around the detected card on the frame using `cv2.rectangle()` and add the card name text above the bounding box using `cv2.putText()`.

#### 6.5: Draw the borderline and display card values

```python
# Draw the borderline on the screen
cv2.line(frame, (640, 0), (640, 720), (255, 0, 0), 2)

# Display the latest card values on the frame
for i, (player, card_value) in enumerate(latest_card_values.items()):
    cv2.putText(frame, f"{player}: {card_value}", (10 + (i * 640), 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
```
After processing all the detected cards, we draw a borderline on the screen to separate the player regions using `cv2.line()`. We also display the latest card values for each player on the frame using `cv2.putText()`. We iterate over the `latest_card_values` dictionary and position the text accordingly.

#### 6.6: Compare the card values and display the result

```python
# Compare the card values and display the player with the bigger number
player1_card = latest_card_values["Player 1"]
player2_card = latest_card_values["Player 2"]

if player1_card.isdigit() and player2_card.isdigit():
    if int(player1_card) > int(player2_card):
        result_text = "Player 1 has the bigger number"
    elif int(player2_card) > int(player1_card):
        result_text = "Player 2 has the bigger number"
    else:
        result_text = "Both players have the same number"
else:
    result_text = "Cannot compare non-numeric cards"

cv2.putText(frame, result_text, (320, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

In this step, we compare the latest card values played by each player. We retrieve the card values for "Player 1" and "Player 2" from the `latest_card_values` dictionary. If both card values are numeric (checked using `isdigit()`), we compare them using `int()` to determine which player has the bigger number. We store the result in the `result_text` variable. If the card values are not numeric, we display a message indicating that comparison is not possible. Finally, we display the result text on the frame using `cv2.putText()`.

#### 6.7: Display the annotated frame and handle key events

```python
# Display the annotated frame
cv2.imshow("UNO YOLOv8 OBB", frame)

# Break the loop if 'q' is pressed
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

We display the annotated frame using `cv2.imshow()`, giving it a window name of "UNO YOLOv8 OBB". We also check for key events using `cv2.waitKey(1)`. If the 'q' key is pressed, we break out of the video processing loop.

### Step 7: Release resources and close windows

```python
cap.release()
cv2.destroyAllWindows()
```

Finally, release the video capture object and close all OpenCV windows.

And that's it! You now have a program that detects UNO cards using YOLOv8 and OpenCV. It processes a video stream, detects the cards played by each player, and displays the player with the bigger number card.

Feel free to modify the code to add more functionality or customize it according to your specific requirements.

Happy UNO card detection!
