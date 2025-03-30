import cv2
import os
import logging
import numpy as np
from deepface import DeepFace
from datetime import datetime
import tempfile
from concurrent.futures import ThreadPoolExecutor
import io
import torch
from ultralytics import YOLO
import concurrent.futures
import json

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model and set the device dynamically
yolo_model = YOLO('Data/yolov11n-face.pt')
yolo_model.to(device)  # Move model to 'cuda' or 'cpu'

# Configuration Constants
THRESHOLD = 0.7
MODEL_NAME = 'VGG-Face'
MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 50

DATABASE_PATH = 'Data/Faces'
MAIN_DIR = 'Data'

for directory in [MAIN_DIR, DATABASE_PATH]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure necessary directories exist
os.makedirs('Data/Output/Frame/Known_Faces', exist_ok=True)
os.makedirs('Data/Output/Frame/Unknown_Faces', exist_ok=True)
os.makedirs('Data/Output/JSON', exist_ok=True)
os.makedirs('Data/Output/Log', exist_ok=True)

# Logging setup
logging.basicConfig(filename='Data/Output/Log/Video_Processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_database(db_path):
    """Load face database and return the path for DeepFace."""
    return db_path

# Load face database
face_db = load_database(DATABASE_PATH)  # Load database embeddings once

def process_face(temp_face_path, coord, db_path, threshold=THRESHOLD):
    """Process individual face image and return the result."""
    try:
        # Perform face recognition with DeepFace
        result_df = DeepFace.find(img_path=temp_face_path, db_path=db_path, model_name=MODEL_NAME, enforce_detection=False, threshold=threshold)
        
        # Check if DeepFace returned valid results
        if not result_df or result_df[0].empty:
            logging.warning(f"No results found for face at {coord} (image path: {temp_face_path})")
            return coord, "Unknown"
        
        # Extract the distance column to verify identity
        distance_column = [col for col in result_df[0].columns if 'distance' in col]
        
        # If a distance column exists and meets the threshold
        if distance_column and result_df[0].iloc[0][distance_column[0]] <= threshold:
            verified_name = result_df[0].iloc[0]['identity'].split(os.sep)[-2]  # Extract the name from the path
            logging.info(f"Face recognized as {verified_name} at {coord} (image path: {temp_face_path})")
            return coord, verified_name
        else:
            logging.info(f"Face at {coord} (image path: {temp_face_path}) did not meet threshold.")
            return coord, "Unknown"

    except Exception as e:
        # Log error in case of any exception
        logging.error(f"Error processing {temp_face_path} at {coord}: {e}")
        return coord, "Unknown"

# Function for recognizing faces in parallel
def detect_and_recognize_faces(frame, db_path, THRESHOLD):
    """Recognize faces in the given frame using YOLOv11 and DeepFace."""
    results = []
    
    # Define the confidence threshold for YOLOv11
    confidence_threshold = 0.7  # Adjust this value as needed
    
    # Perform YOLOv11 inference on the frame
    detections = yolo_model(frame)  # YOLOv11 inference
    
    face_images = []
    coords = []
    
    try:
        # Iterate over detections and extract valid face crops
        for detection in detections[0].boxes:
            confidence = detection.conf.cpu().numpy()  # Get the confidence score
            
            # Check if detection is valid and confidence is above threshold
            if confidence >= confidence_threshold:
                try:
                    # Extract coordinates (xmin, ymin, xmax, ymax)
                    x_min, y_min, x_max, y_max = detection.xyxy[0].cpu().numpy()

                    # Filter detections based on minimum face size
                    if (x_max - x_min >= MIN_FACE_WIDTH) and (y_max - y_min >= MIN_FACE_HEIGHT):
                        face_crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        face_images.append(face_crop)
                        coords.append((int(x_min), int(y_min), int(x_max), int(y_max)))

                except Exception as e:
                    logging.error(f"Error extracting coordinates from detection: {e}")
                    continue
                
        # Process the face images in parallel if any valid faces are found
        if face_images and coords:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_face, face_crop, coord, db_path, THRESHOLD)
                    for face_crop, coord in zip(face_images, coords)
                ]
                
                # Collect results from parallel processing
                for future in futures:
                    results.append(future.result())

    except Exception as e:
        logging.error(f"Error during face recognition: {e}")

    return results

# Function to capture video from webcam and perform face recognition
def capture_and_process():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        logging.error("Could not open webcam")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

            # Submit face recognition task
            future = executor.submit(detect_and_recognize_faces, frame, face_db, THRESHOLD)
            try:
                results = future.result()  # Get face recognition result
            except Exception as e:
                logging.error(f"Error in face recognition: {e}")
                results = []

            if results:
                json_data = {
                    "Number_of_faces": len(results),
                    "Faces": []
                }

                for coords, identity in results:
                    x_min, y_min, x_max, y_max = coords
                    json_data["Faces"].append({
                        "x_min": x_min, "y_min": y_min,
                        "x_max": x_max, "y_max": y_max,
                        "identity": identity
                    })

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, identity, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Save frame if needed
                    if identity == "Unknown":
                        filename = f"Data/Output/Frame/Unknown_Faces/{timestamp}.png"
                    else:
                        filename = f"Data/Output/Frame/Known_Faces/{timestamp}.png"
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Save JSON file asynchronously
                json_filename = f"Data/Output/JSON/{timestamp}.json"
                with open(json_filename, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)

                # Display frame
                cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_process()