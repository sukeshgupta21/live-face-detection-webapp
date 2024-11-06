import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found")
else:
    print("Camera opened successfully")

