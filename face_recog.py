import cv2
import numpy as np

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the dataset
face_data = []
count = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw the bounding box around the detected faces and save the face data
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face region of interest (ROI)
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Save the face ROI in the dataset
        face_data.append(face_section)
        count += 1

    # Display the output frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit the program and save the dataset
    if cv2.waitKey(1) == ord('q'):
        break

# Convert the face dataset to a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save the face dataset as a numpy array
np.save('face_data.npy', face_data)
print("Dataset saved successfully...")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
