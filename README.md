# Real-Time Face Recognition (Python + OpenCV)

A real-time face recognition system that runs on a PC webcam. It detects faces, identifies enrolled people by name, and labels everyone else as **Unknown**. It can recognize several people at the same time.

## How it works

Webcam -> YuNet face detection -> SFace embedding (128 values) -> cosine similarity with enrolled faces -> decision (name or Unknown)

- **Detection:** YuNet (OpenCV Zoo), lightweight ONNX model
- **Recognition:** SFace (OpenCV Zoo), converts a face into an embedding
- **Matching:** cosine similarity with a threshold (default 0.363)
- Both models are pretrained. The project contributes the pipeline: enrollment, matching, threshold tuning and evaluation (FAR/FRR).

## Project structure

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/manaflaassiri03-lang/face-recognition.git
cd face-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
.\download_models.ps1
```

If PowerShell blocks the activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Python 3.10 to 3.12 is recommended. Linux/Mac: use `source venv/bin/activate` and download the two models manually from the OpenCV Zoo.

## Usage

### 1. Enroll a person

```powershell
python src\enroll.py manaf
```

Look at the camera, press **SPACE** about 10 times (vary angle and light), then **Q**. Repeat with another name to enroll more people:

```powershell
python src\enroll.py sara
```

### 2. Run live recognition

```powershell
python src\main.py
```

- Green box + name: recognized person
- Red box + "Unknown": not enrolled
- Press **Q** to quit

### 3. Evaluate accuracy (optional)

```powershell
python src\evaluate.py genuine manaf 100     # you in front of the camera
python src\evaluate.py impostor manaf 100    # other people (or phone photos)
python src\thresholds.py                     # FRR / FAR table
```

Then set the best value in `COSINE_THRESHOLD` inside `src/main.py`.

### Manage enrolled people

```powershell
Get-ChildItem data\enrolled                  # list
Remove-Item data\enrolled\sara.npy           # delete one person
```

## Results

| Threshold | FRR % | FAR % |
|---|---|---|
| 0.30 | | |
| 0.363 | | |
| 0.45 | | |

Average speed on laptop CPU: ___ FPS. (Fill in with your own measurements.)

## Limitations

- No liveness detection: a photo of an enrolled person shown on a phone can be accepted.
- Accuracy drops with strong side angles, low light and masks.
- Not suitable for security-critical use.

## Privacy and GDPR

- Only enroll people who gave their consent.
- The system stores embeddings, not photos, but embeddings are still biometric data.
- The `data/` folder is excluded from Git. Delete a person's `.npy` file to remove their data.

## Future work

- Stability filter and CSV logging
- Blink detection (liveness)
- Optimization (smaller input, INT8 quantization) and deployment on an embedded board with a camera
- Integration with CAN/GPIO to trigger an action on recognition

## Credits

Models from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet, SFace).

## Stability filter and logging

- Each face is tracked across frames. A name is confirmed only when it wins at least 5 of the last 8 frames, which removes label flickering (yellow = checking, green = recognized, red = unknown).
- Every confirmed recognition is logged to `data/log.csv` (date, time, name, score) with a 10-second cooldown per person.
