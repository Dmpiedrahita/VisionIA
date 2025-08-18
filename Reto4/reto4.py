import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access the camera")
    exit()

colorRanges = {
    "Yellow": (np.array([15, 80, 180]), np.array([35, 255, 255])),
    "Pink":   (np.array([160, 120, 120]), np.array([179, 255, 255])),
    "Light Green": (np.array([35, 80, 80]), np.array([65, 255, 255])),
    "Yellow": (np.array([5, 40, 160]), np.array([20, 120, 255])),
    "Purple": (np.array([125, 100, 100]), np.array([150, 255, 255]))
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for colorName, (lower, upper) in colorRanges.items():
        mask = cv2.inRange(hsv, lower, upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # filter noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, colorName, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
