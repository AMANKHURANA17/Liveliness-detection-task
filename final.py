import cv2
import dlib
import numpy as np
import time

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate the EAR (Eye Aspect Ratio)
threshold_value=8
liv_score=0

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect smile
def detect_smile(shape):
    left_corner = shape[49]
    right_corner = shape[55]
    top_lip = shape[52]
    bottom_lip = shape[58]
    
    horizontal_dist = np.linalg.norm(right_corner - left_corner)
    vertical_dist = np.linalg.norm(top_lip - bottom_lip)
    
    smile_ratio = horizontal_dist / vertical_dist
    return smile_ratio > 1.65  # Threshold for smile

# Capture video frame (example with webcam)
cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) > 1:
        print(f"Exiting: Number of faces detected is {len(faces)}, which is not equal to 1")
        break
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get the landmarks/parts for the face
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Draw the landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Detect smile
        if detect_smile(shape):
            #cv2.putText(frame, "Smiling", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            liv_score+=1
            
        
        # Get the coordinates for the left and right eye
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        # Calculate the EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        # Threshold for detecting a blink (typically 0.2 is used)
        ear_threshold = 0.15
        if left_ear < ear_threshold or right_ear < ear_threshold:
            #cv2.putText(frame, "Winking", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            liv_score+=1

    # Display the frame with detected face(s) and landmarks
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the elapsed time is more than 5 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        if liv_score < threshold_value:
            print(f"Exiting: Liveness score {liv_score} did not reach threshold {threshold_value} within 5 seconds")
            break
        else:
            print(f"Success: Liveness score {liv_score} reached threshold value")
            #print(threshold_value)
            break
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()