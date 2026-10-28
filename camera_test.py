import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

print("Camera started! Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read camera.")
        break

    cv2.imshow("Human Recognition - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()