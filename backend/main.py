from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5173",  # Update with your frontend's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the YOLOv8 model
model = YOLO('uno-obb.pt')

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

# Define the request and response models
class DetectionResult(BaseModel):
    class_name: str
    score: float
    bbox: list

"""
    _summary_: Health check endpoint
    _description_: This endpoint performs a simple inference to check if the model is loaded and functioning properly.
    _request_body_: None
    Returns:
        _type_: _description_
"""
@app.get("/")
async def health_check():
    try:
        # Perform a simple inference to check if the model is loaded and functioning
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image)
        
        if results is not None:
            return {"status": "healthy", "message": "YOLO model is loaded and functioning properly"}
        else:
            return {"status": "unhealthy", "message": "YOLO model inference returned None"}
    
    except Exception as e:
        return {"status": "unhealthy", "message": f"Error occurred during health check: {str(e)}"}


"""
    _summary_: Detect UNO cards in an image
    _description_: This endpoint takes a base64 image as input and returns the detected UNO cards along with their bounding boxes and confidence scores.
    _request_body_(JSON):
        image: str - The input image in base64 format
    Returns:
        _type_: list[DetectionResult]
        _description_: A list of DetectionResult objects containing the detected UNO cards along with their bounding boxes and confidence scores
"""
@app.post("/detect", response_model=list[DetectionResult])
   
async def detect_uno_cards(request: Request):
    data = await request.json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference on the frame
    results = model(frame)[0]

    detection_results = []
    for result in results:
        if result is not None:
            # Extract the detected class and bounding box
            obb = result.obb
            detected_classes = obb.cls

            # Iterate over the detected classes
            for cls in detected_classes:
                card = UNO[int(cls)]

                # Get the bounding box coordinates
                bbox = obb.xyxy[0].cpu().numpy().tolist()

                # Get the confidence score
                score = float(obb.conf)

                # Create a DetectionResult object
                detection_result = DetectionResult(
                    class_name=str(card),
                    score=score,
                    bbox=bbox
                )

                # only append the detection result if the score is above a certain threshold
                if score > 0.7:
                    detection_results.append(detection_result)

    return detection_results