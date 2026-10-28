import cv2, numpy as np, glob, os, time, csv
from collections import deque, Counter
from datetime import datetime

# ---------------- Settings ----------------
COSINE_THRESHOLD = 0.363   # similarity needed to accept a name
HISTORY = 8                # frames kept per tracked face
CONFIRM = 5                # a name must win this many of those frames
MAX_DIST = 120             # max pixel distance to link a face to a track
MAX_MISSED = 10            # frames before a lost track is deleted
LOG_COOLDOWN = 10          # seconds between two log lines for the same person
LOG_PATH = "data/log.csv"
# ------------------------------------------

detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

db = {}
for path in glob.glob("data/enrolled/*.npy"):
    db[os.path.splitext(os.path.basename(path))[0]] = np.load(path)
print("Enrolled:", list(db.keys()))

# CSV log
os.makedirs("data", exist_ok=True)
new_file = not os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="", encoding="utf-8")
log = csv.writer(log_file)
if new_file:
    log.writerow(["date", "time", "name", "score"])
last_logged = {}   # name -> last log time


def best_match(feat):
    best_name, best_score = "Unknown", 0.0
    for name, embs in db.items():
        for e in embs:
            s = recognizer.match(feat, e.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE)
            if s > best_score:
                best_score, best_name = s, name
    if best_score < COSINE_THRESHOLD:
        best_name = "Unknown"
    return best_name, best_score


def stable_label(track):
    """Return (label, avg_score, confirmed) from the track history."""
    hist = track["history"]
    if len(hist) < CONFIRM:
        return "Checking...", 0.0, False
    counts = Counter(n for n, _ in hist)
    name, n = counts.most_common(1)[0]
    if n < CONFIRM:
        return "Checking...", 0.0, False
    scores = [s for nm, s in hist if nm == name]
    return name, float(np.mean(scores)), True


def write_log(name, score):
    now = time.time()
    if now - last_logged.get(name, 0) < LOG_COOLDOWN:
        return
    last_logged[name] = now
    t = datetime.now()
    log.writerow([t.strftime("%Y-%m-%d"), t.strftime("%H:%M:%S"), name, f"{score:.3f}"])
    log_file.flush()
    print(f"[LOG] {t.strftime('%H:%M:%S')}  {name}  {score:.2f}")


tracks = {}
next_id = 0
cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    used = set()
    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            center = (x + fw / 2, y + fh / 2)

            # link this face to the nearest existing track
            best_id, best_d = None, MAX_DIST
            for tid, t in tracks.items():
                if tid in used:
                    continue
                d = np.hypot(center[0] - t["center"][0], center[1] - t["center"][1])
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = next_id
                next_id += 1
                tracks[best_id] = {"history": deque(maxlen=HISTORY), "center": center,
                                   "missed": 0, "last_name": None}
            used.add(best_id)
            t = tracks[best_id]
            t["center"] = center
            t["missed"] = 0

            # recognize this frame and store the result
            feat = recognizer.feature(recognizer.alignCrop(frame, face))
            t["history"].append(best_match(feat))

            label, score, confirmed = stable_label(t)
            if not confirmed:
                color = (0, 255, 255)          # yellow
            elif label == "Unknown":
                color = (0, 0, 255)            # red
            else:
                color = (0, 255, 0)            # green

            if confirmed:
                write_log(label, score)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            text = label if not confirmed else f"{label} {score:.2f}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # remove lost tracks
    for tid in list(tracks.keys()):
        if tid not in used:
            tracks[tid]["missed"] += 1
            if tracks[tid]["missed"] > MAX_MISSED:
                del tracks[tid]

    now = time.time()
    cv2.putText(frame, f"FPS: {1/max(now-prev,1e-6):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    prev = now

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
