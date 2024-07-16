import cv2

# Load pre-trained Haar Cascade classifiers for face, smile, and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)  

# Preset threshold for liveness score
LIVENESS_THRESHOLD = 2 
blink = False
detect = False
def check_liveness(roi_gray):
    global detect, blink,liveness_score
    # Detect eyes and smiles within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))

    liveness_score = 0
    if len(smiles) >0:
        liveness_score += 1

    if detect == True and len(eyes) == 0:
        blink = True
        #print('eye blinked')
    elif len(eyes)>0:
        if not blink:
            detect=True


    if blink == True:
        liveness_score += 1
        blink=False

    return liveness_score


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Check the number of detected faces
    if len(faces) > 1:
        print(f"Number of faces detected: {len(faces)}. Exiting.")
        break


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Check liveness score
        liveness_score = check_liveness(roi_gray)
        print(f"Liveness score: {liveness_score}")

    
        if liveness_score >= LIVENESS_THRESHOLD:
            cv2.putText(frame, 'Success', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        '''
        else:
            cv2.putText(frame, 'Liveness Check Failed', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Liveness score below threshold. Exiting.")
            break
        '''

    cv2.imshow('Liveness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested by user.")
        break

cap.release()
cv2.destroyAllWindows()

