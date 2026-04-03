import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    frames.append(img)

    if len(frames) == 10:
        # التوقع العشوائي بدل الموديل الحقيقي
        prediction = np.random.choice(["A", "B"])
        cv2.putText(frame, f"Sign: {prediction}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        frames = []

    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()