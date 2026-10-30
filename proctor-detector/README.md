# Proctor Detector (Phase 1: phone detection)

Reads a video source, detects people + phones with YOLO, and reports flags
(plus a short evidence clip) to `proctor-backend`.

## Run
```bash
cp .env.example .env          # set SESSION_ID to a session created in the backend
uv sync
uv run detector --show                          # webcam
uv run detector --source videos/test.mp4 --session 1 --show
```
The backend must be running and the session must exist (`POST /sessions`).
First run downloads `yolo11n.pt` automatically.

## How a flag is decided
1. YOLO (COCO "cell phone" class) + tracker runs on each frame.
2. Each phone is assigned to the nearest/containing person track.
3. A student is flagged only if a phone is seen in `CONFIRM_FRAMES` of the last `WINDOW_FRAMES`
   processed frames (default 8 of 15). This removes one-frame false detections.
4. After a flag, the same student is ignored for `COOLDOWN_SECONDS`.
5. A clip (4 s before + 3 s after) is saved and uploaded to the alert.

Tune with env vars: `PHONE_CONF`, `WINDOW_FRAMES`, `CONFIRM_FRAMES`, `COOLDOWN_SECONDS`,
`FRAME_SKIP`, `DEVICE`.

## Notes
- Pretrained COCO weights miss phones that are small, on laps or tilted. Fine-tuning on your own
  exam-hall footage (next step) is how recall improves.
- Windows + NVIDIA GPU: install the CUDA build of PyTorch, then set `DEVICE=0`.

## Using a real exam-hall camera
1. **Test the URL in VLC first** (Media > Open Network Stream). If VLC can't play it, neither can the detector.
2. Put the URL in `.env` as `SOURCE=rtsp://...`. Use the camera's **main stream** (full resolution): phones are tiny.
   Never commit `.env`, since it contains the camera password.
3. Run `uv run detector --show`. The camera must be reachable from the PC (same network/VLAN).

The stream reader always uses the newest frame (no growing lag) and reconnects automatically if the
camera or network drops.

### Collect training frames from the hall (do this before the real exam)
```bash
uv run detector-grab --every 5 --out dataset/images
```
Record mock exam sessions with volunteers using phones in realistic ways (desk, lap, under desk, earbuds),
then label these frames and fine-tune the model. Footage from the *actual* camera angle is what
makes detection reliable.

### Camera placement
- Mount high at the front corner, angled down so faces and desks of every row are visible.
- 1080p minimum; 4K cameras help a lot for the back rows.
- Even lighting, no bright windows in the frame.
- Speed: use a GPU (`DEVICE=0`). On CPU, raise `FRAME_SKIP` or lower `IMGSZ`.
