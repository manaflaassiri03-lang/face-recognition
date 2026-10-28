import cv2, numpy as np, glob, os, time

COSINE_THRESHOLD = 0.363   # adjust after evaluation

detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

db = {}
for path in glob.glob("data/enrolled/*.npy"):
    db[os.path.splitext(os.path.basename(path))[0]] = np.load(path)
print("Enrolled:", list(db.keys()))

cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            feat = recognizer.feature(recognizer.alignCrop(frame, face))

            best_name, best_score = "Unknown", 0.0
            for name, embs in db.items():
                for e in embs:
                    score = recognizer.match(feat, e.reshape(1, -1),
                                             cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score, best_name = score, name

            if best_score < COSINE_THRESHOLD:
                best_name = "Unknown"

            x, y, fw, fh = face[:4].astype(int)
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(frame, f"{best_name} {best_score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    now = time.time()
    cv2.putText(frame, f"FPS: {1/(now-prev):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    prev = now

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
