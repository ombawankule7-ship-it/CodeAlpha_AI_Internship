import cv2
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


# Open webcam
cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        break

    # Detect and track objects
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Draw boxes and IDs
    output = results[0].plot()

    # Display
    cv2.imshow(
        "CodeAlpha Object Detection and Tracking",
        output
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindow()