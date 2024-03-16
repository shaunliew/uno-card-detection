import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('uno-obb.pt')

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Define the UNO card classes
UNO = {
    0: 0,
    1: 1,
    2: "+4",
    3: "+2",
    4: "reverse",
    5: "skip",
    6: "wild",
    7: 2,
    8: 3,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 9
}

# Define the player regions
player_regions = {
    "Player 1": (0, 0, 640, 720),
    "Player 2": (640, 0, 1280, 720)
}

# Initialize the player hands
player_hands = {
    "Player 1": [],
    "Player 2": []
}

# Initialize the latest card values
latest_card_values = {
    "Player 1": "",
    "Player 2": ""
}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)[0]
        for result in results:
            if result is not None:
                # Extract the detected class and bounding box
                obb = result.obb
                detected_classes = obb.cls

                # Iterate over the detected classes
                for cls in detected_classes:
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
                        cv2.putText(frame, str(card), (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw the borderline on the screen
        cv2.line(frame, (640, 0), (640, 720), (255, 0, 0), 2)

        # Display the latest card values on the frame
        for i, (player, card_value) in enumerate(latest_card_values.items()):
            cv2.putText(frame, f"{player}: {card_value}", (10 + (i * 640), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

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

        cv2.putText(frame, result_text, (320, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("UNO YOLOv8 OBB", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()