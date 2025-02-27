import cv2
import numpy as np

# Load the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained age detection model
age_model = "age_deploy.prototxt"
age_weights = "age_net.caffemodel"
age_net = cv2.dnn.readNet(age_model, age_weights)

# Age ranges for the model
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face ROI
        face = frame[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean, swapRB=False)

        # Predict age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Display the age prediction
        text = f'Age: {age}'
        y_offset = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the output frame
    cv2.imshow("Age Prediction", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
