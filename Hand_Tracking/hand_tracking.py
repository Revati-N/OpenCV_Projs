import cv2 # Import OpenCV for image processing
import mediapipe as mp # Import MediaPipe for hand tracking
import time

cap = cv2.VideoCapture(0) # Use 0 for the default camera, or replace with external camera index (1, 2, etc.) if needed

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)