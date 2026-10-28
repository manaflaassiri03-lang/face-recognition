import cv2, numpy as np, glob, os, sys, time, csv
import psutil

N = 100                                   # frames per test
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = default, 1 = single core
RESOLUTIONS = [(640, 480), (480, 360), (320, 240)]

if THREADS > 0:
    cv2.setNumThreads(THREADS)

DET = "models/face_detection_yunet_2023mar.onnx"
REC = "models/face_recognition_sface_2021dec.onnx"
detector = cv2.FaceDetectorYN.create(DET, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(REC, "")

# enrolled embeddings (for realistic matching cost)
db = [np.load(p) for p in glob.glob("data/enrolled/*.npy")]
embs = np.concatenate(db) if db else np.zeros((1, 128), dtype=np.float32)

proc = psutil.Process(os.getpid())
print(f"Threads: {'default (' + str(cv2.getNumThreads()) + ')' if THREADS == 0 else THREADS}")
print(f"Model sizes: detector {os.path.getsize(DET)/1e6:.2f} MB, recognizer {os.path.getsize(REC)/1e6:.2f} MB")
print("Look at the camera during the test (a face must be visible).\n")

cap = cv2.VideoCapture(0)
results = []

for (rw, rh) in RESOLUTIONS:
    t_det, t_emb, t_match, t_tot = [], [], [], []
    got = 0
    while got < N:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (rw, rh))
        detector.setInputSize((rw, rh))

        t0 = time.perf_counter()
        _, faces = detector.detect(frame)
        t1 = time.perf_counter()
        if faces is None:
            cv2.imshow("Benchmark", frame)
            cv2.waitKey(1)
            continue

        aligned = recognizer.alignCrop(frame, faces[0])
        feat = recognizer.feature(aligned)
        t2 = time.perf_counter()
        for e in embs:
            recognizer.match(feat, e.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE)
        t3 = time.perf_counter()

        t_det.append((t1 - t0) * 1000)
        t_emb.append((t2 - t1) * 1000)
        t_match.append((t3 - t2) * 1000)
        t_tot.append((t3 - t0) * 1000)
        got += 1

        cv2.putText(frame, f"{rw}x{rh}  {got}/{N}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Benchmark", frame)
        cv2.waitKey(1)

    if t_tot:
        total = np.mean(t_tot[10:])      # skip warm-up frames
        results.append((f"{rw}x{rh}", np.mean(t_det[10:]), np.mean(t_emb[10:]),
                        np.mean(t_match[10:]), total, 1000 / total))

cap.release()
cv2.destroyAllWindows()

ram = proc.memory_info().rss / 1e6
print("Resolution | Detect ms | Embed ms | Match ms | Total ms |  FPS")
for r in results:
    print(f"{r[0]:>10} | {r[1]:9.1f} | {r[2]:8.1f} | {r[3]:8.2f} | {r[4]:8.1f} | {r[5]:5.1f}")
print(f"\nProcess RAM: {ram:.0f} MB   (enrolled embeddings: {len(embs)})")

os.makedirs("data", exist_ok=True)
tag = "default" if THREADS == 0 else f"{THREADS}thread"
with open(f"data/benchmark_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["resolution", "detect_ms", "embed_ms", "match_ms", "total_ms", "fps"])
    for r in results:
        w.writerow([r[0]] + [f"{v:.2f}" for v in r[1:]])
print(f"Saved data/benchmark_{tag}.csv")
