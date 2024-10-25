from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import mediapipe as mp
import numpy as np
import uuid
from fastapi.responses import FileResponse
import json

#mp_pose = mp.solutions.pose
app = FastAPI()

UPLOAD_DIRECTORY = "./uploaded_videos"
PROCESSED_DIRECTORY = "./processed_data"


os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}


@app.get("/processed_data/{data_id}")
def get_processed_data(data_id: str):
    keypoints_file = os.path.join(PROCESSED_DIRECTORY, f"{data_id}_keypoints.npy")
    if not os.path.exists(keypoints_file):
        raise HTTPException(status_code=404, detail="Data not found.")

    return FileResponse(keypoints_file, media_type='application/octet-stream', filename=f"{data_id}_keypoints.npy")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    extension = filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    with open(file_location, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    try:
        keypoints_data = process_video(file_location)
        # Save keypoints data as a NumPy file
        keypoints_file = os.path.join(PROCESSED_DIRECTORY, f"{filename}_keypoints.npy")
        np.save(keypoints_file, keypoints_data)

        # Analyze movements
        analysis_results = analyze_movement(keypoints_data)
        analysis_file = os.path.join(PROCESSED_DIRECTORY, f"{filename}_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

    data_id = f"{filename}_{uuid.uuid4().hex}"
    keypoints_file = os.path.join(PROCESSED_DIRECTORY, f"{data_id}_keypoints.npy")
    np.save(keypoints_file, keypoints_data)
        

    return {
        "filename": filename,
        "message": "File uploaded and processed successfully.",
        "data_id": data_id
    }


def process_video(video_path):
    # Initialize variables
    keypoints_all_frames = []
    mp_pose = mp.solutions.pose
    frame_index = 0

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or fps is None:
        fps = 30.0  # Set a default FPS value if unable to get from the video

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for '.avi' files
    output_video_path = os.path.join(UPLOAD_DIRECTORY, f"{os.path.basename(video_path)}_processed.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise Exception("Could not open VideoWriter")

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and detect pose
            results = pose.process(image)

            # Extract keypoints
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Write the frame to the output video
            try:
                out.write(frame)
            except Exception as e:
                print(f"Error writing frame {frame_index}: {e}")

            # Extract keypoints
            if results.pose_landmarks:
                keypoints = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]
                keypoints_all_frames.append(keypoints)
            else:
                keypoints_all_frames.append(np.full((33, 4), np.nan, dtype=np.float32))

            frame_index += 1

    cap.release()
    out.release()

    num_frames = len(keypoints_all_frames)
    num_landmarks = 33
    keypoints_array = keypoints_array = np.zeros((num_frames, num_landmarks, 4), dtype=np.float32)


    for i, keypoints in enumerate(keypoints_all_frames):
        if keypoints is not None:
            keypoints_array[i] = keypoints
        else:
            keypoints_array[i] = np.full((num_landmarks, 4), np.nan, dtype=np.float32)  # or leave zeros if appropriate

    # Convert to NumPy array
    keypoints_array = np.array(keypoints_all_frames, dtype=object)

   

    

    return keypoints_array


def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three points a, b, and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors BA and BC
    ba = a - b
    bc = c - b
    
    # Calculate the angle in radians
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    return angle_degrees


def analyze_elbow_angles(keypoints_frame):
    """
    Analyzes elbow angles for a single frame.
    Returns a dictionary with elbow angles.
    """
    landmarks = keypoints_frame
    
    # Check if landmarks are valid
    if np.isnan(landmarks).all():
        return {'left_elbow_angle': None, 'right_elbow_angle': None}
    
    # Get the required landmarks
    left_shoulder = landmarks[11][:2]
    left_elbow = landmarks[13][:2]
    left_wrist = landmarks[15][:2]
    
    right_shoulder = landmarks[12][:2]
    right_elbow = landmarks[14][:2]
    right_wrist = landmarks[16][:2]
    
    # Calculate angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    return {
        'left_elbow_angle': left_elbow_angle,
        'right_elbow_angle': right_elbow_angle
    }

def analyze_weight_distribution(keypoints_frame):
    """
    Analyzes weight distribution for a single frame.
    Returns a string indicating weight distribution.
    """
    landmarks = keypoints_frame
    
    # Check if landmarks are valid
    if np.isnan(landmarks).all():
        return {'weight_distribution': None}
    
    left_hip = landmarks[23][:2]
    right_hip = landmarks[24][:2]
    
    # Calculate the center point between hips
    center_x = (left_hip[0] + right_hip[0]) / 2
    
    # Define thresholds for leaning
    # Assuming x ranges from 0 (left) to 1 (right)
    if center_x < 0.45:
        weight_distribution = 'Leaning Left'
    elif center_x > 0.55:
        weight_distribution = 'Leaning Right'
    else:
        weight_distribution = 'Centered'
    
    return {'weight_distribution': weight_distribution}

def analyze_movement(keypoints_array):
    """
    Analyzes movements across all frames.
    Returns a list of analysis results for each frame.
    """
    analysis_results = []
    
    for frame_index, keypoints_frame in enumerate(keypoints_array):
        if np.isnan(keypoints_frame).all():
            analysis_results.append({
                'frame': frame_index,
                'left_elbow_angle': None,
                'right_elbow_angle': None,
                'weight_distribution': None,
                'feedback': 'No pose detected'
            })
            continue
        
        elbow_angles = analyze_elbow_angles(keypoints_frame)
        weight_dist = analyze_weight_distribution(keypoints_frame)
        
        feedback = []
        
        # Analyze elbow angles
        if elbow_angles['left_elbow_angle'] is not None:
            if elbow_angles['left_elbow_angle'] < 160:
                feedback.append('Left arm is bent')
            else:
                feedback.append('Left arm is straight')
        
        if elbow_angles['right_elbow_angle'] is not None:
            if elbow_angles['right_elbow_angle'] < 160:
                feedback.append('Right arm is bent')
            else:
                feedback.append('Right arm is straight')
        
        # Analyze weight distribution
        if weight_dist['weight_distribution'] is not None:
            feedback.append(f"Weight is {weight_dist['weight_distribution']}")
        
        analysis_results.append({
            'frame': frame_index,
            'left_elbow_angle': elbow_angles['left_elbow_angle'],
            'right_elbow_angle': elbow_angles['right_elbow_angle'],
            'weight_distribution': weight_dist['weight_distribution'],
            'feedback': ', '.join(feedback)
        })
    
    return analysis_results
