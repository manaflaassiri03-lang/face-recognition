import cv2, numpy as np, sys, os

name = sys.argv[1] if len(sys.argv) > 1 else "me"

detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

cap = cv2.VideoCapture(0)
embeddings = []
print("Press SPACE to capture (10 shots, vary angle and light). Press Q to finish.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        x, y, fw, fh = faces[0][:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {len(embeddings)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Enroll", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and faces is not None:
        aligned = recognizer.alignCrop(frame, faces[0])
        embeddings.append(recognizer.feature(aligned)[0])
        print("Captured", len(embeddings))
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if embeddings:
    os.makedirs("data/enrolled", exist_ok=True)
    np.save(f"data/enrolled/{name}.npy", np.array(embeddings))
    print(f"Saved {len(embeddings)} embeddings for {name}")
