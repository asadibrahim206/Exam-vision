import argparse
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass

import cv2
import httpx
from ultralytics import YOLO

from .api_client import ApiClient
from .clips import resize_to_width, write_clip
from .config import Settings
from .stream import VideoStream

PERSON, PHONE = 0, 67  # COCO class ids ("cell phone" = 67)


@dataclass
class PendingClip:
    alert_id: int
    frames: list      # [(t, frame), ...]
    end_t: float


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def assign_owner(phone_box, persons):
    """Tracker id of the person this phone most likely belongs to (-1 if unknown)."""
    cx, cy = _center(phone_box)
    best_id, best_d = -1, math.inf
    for pid, pb in persons:
        pcx, pcy = _center(pb)
        inside = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
        d = math.hypot(cx - pcx, cy - pcy)
        if not inside and d > (pb[3] - pb[1]):
            continue
        if d < best_d:
            best_id, best_d = (pid if pid is not None else -1), d
    return best_id


def finish_clip(pc: PendingClip, api: ApiClient) -> None:
    frames = [f for _, f in pc.frames]
    duration = pc.frames[-1][0] - pc.frames[0][0] if len(pc.frames) > 1 else 0
    fps = min(30.0, max(1.0, len(frames) / duration)) if duration > 0 else 10.0
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        if write_clip(frames, fps, path):
            api.upload_clip(pc.alert_id, path)
            print(f"  clip uploaded for alert {pc.alert_id}")
        else:
            print(f"  could not encode clip for alert {pc.alert_id}")
    except httpx.HTTPError as e:
        print(f"  clip upload failed for alert {pc.alert_id}: {e}")
    finally:
        os.remove(path)


def run(s: Settings) -> None:
    stream = VideoStream(s.source)
    model = YOLO(s.model)
    api = ApiClient(s.api_url, s.api_key)

    buffer: deque = deque()            # (t, small_frame) covering the last pre_seconds
    hits: dict[int, deque] = {}        # owner id -> recent "phone seen" flags
    last_alert: dict[int, float] = {}  # owner id -> time of last alert
    pending: list[PendingClip] = []
    t0, idx = None, 0

    print(f"Running on {s.source} (live={stream.live}), session {s.session_id}. Ctrl+C to stop.")
    try:
        while True:
            item = stream.read()
            if item is None:
                break
            t, frame = item
            t0 = t if t0 is None else t0
            idx += 1

            small = resize_to_width(frame, s.clip_width)
            buffer.append((t, small))
            while buffer and t - buffer[0][0] > s.pre_seconds:
                buffer.popleft()
            for pc in pending:
                pc.frames.append((t, small))
            for pc in [p for p in pending if t >= p.end_t]:
                finish_clip(pc, api)
                pending.remove(pc)

            if idx % s.frame_skip:
                continue

            res = model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON, PHONE],
                conf=0.25, imgsz=s.imgsz, device=s.device, verbose=False,
            )[0]

            persons, phones = [], []
            for b in res.boxes:
                cls, conf = int(b.cls), float(b.conf)
                box = b.xyxy[0].tolist()
                tid = int(b.id) if b.id is not None else None
                if cls == PERSON:
                    persons.append((tid, box))
                elif cls == PHONE and conf >= s.phone_conf:
                    phones.append((conf, box))

            seen: dict[int, tuple[float, list]] = {}
            for conf, box in phones:
                owner = assign_owner(box, persons)
                if owner not in seen or conf > seen[owner][0]:
                    seen[owner] = (conf, box)

            for owner in seen:
                hits.setdefault(owner, deque(maxlen=s.window_frames))
            for owner, window in list(hits.items()):
                window.append(owner in seen)
                if len(window) == window.maxlen and not any(window):
                    del hits[owner]
                    continue
                if sum(window) >= s.confirm_frames and t - last_alert.get(owner, -math.inf) >= s.cooldown_seconds:
                    last_alert[owner] = t
                    conf, box = seen.get(owner, (0.0, []))
                    print(f"[{t - t0:7.1f}s] PHONE flagged (student track {owner}, conf {conf:.2f})")
                    try:
                        alert = api.create_alert({
                            "session_id": s.session_id,
                            "track_id": owner if owner != -1 else None,
                            "behavior_type": "phone",
                            "confidence": round(conf, 3),
                            "video_ts_seconds": round(t - t0, 2),
                            "details": {"bbox": [round(v, 1) for v in box], "frames_hit": int(sum(window))},
                        })
                        pending.append(PendingClip(alert["id"], list(buffer), t + s.post_seconds))
                    except httpx.HTTPError as e:
                        print(f"  failed to send alert: {e}")

            if s.show:
                cv2.imshow("proctor-detector", resize_to_width(res.plot(), 1280))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for pc in pending:
            finish_clip(pc, api)
        stream.close()
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Exam hall phone detector")
    p.add_argument("--source", help="rtsp url, video file, or webcam index")
    p.add_argument("--session", type=int, help="exam session id from the backend")
    p.add_argument("--show", action="store_true", help="display annotated video")
    args = p.parse_args()

    s = Settings()
    if args.source:
        s.source = args.source
    if args.session:
        s.session_id = args.session
    if args.show:
        s.show = True
    run(s)


if __name__ == "__main__":
    main()
