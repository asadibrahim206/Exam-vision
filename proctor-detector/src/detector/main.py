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
from .behaviors import HeadTurnRule, Person, PhoneRule
from .clips import resize_to_width, write_clip
from .config import Settings
from .stream import VideoStream

PHONE = 67  # COCO "cell phone"


@dataclass
class PendingClip:
    alert_id: int
    frames: list      # [(t, frame), ...]
    end_t: float


def extract_persons(res) -> list[Person]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(boxes)
    kp = res.keypoints
    kxy = kp.xy.cpu().numpy() if kp is not None and kp.xy is not None else None
    kc = kp.conf.cpu().numpy() if kp is not None and kp.conf is not None else None
    return [
        Person(ids[i], b.tolist(), kxy[i] if kxy is not None else None, kc[i] if kc is not None else None)
        for i, b in enumerate(boxes)
    ]


def extract_phones(res, min_conf: float) -> list[tuple[float, list]]:
    return [(float(b.conf), b.xyxy[0].tolist()) for b in res.boxes if float(b.conf) >= min_conf]


def _label(view, text, x, y, fs, color):
    """Text on a solid dark box so it stays readable on any background."""
    th = max(1, int(round(fs * 1.6)))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(view, (x - 4, y - h - 5), (x + w + 4, y + base + 3), (0, 0, 0), -1)
    cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def draw_status(view, persons, phones, head_rule, seconds_needed):
    """Overlay live detector state so you can see WHY something is or isn't flagged."""
    fs = max(0.6, view.shape[1] / 1500)
    for p in persons:
        if p.id is None:
            continue
        d = head_rule.debug.get(p.id) if head_rule is not None else None
        if d is None:
            continue
        if d["turn"] is None:
            text, color = "no face points", (200, 200, 200)
        elif d["nbr"] is None:
            text, color = f"head {d['turn']:+.2f} (not turned)", (255, 255, 255)
        elif not d["nbr"]:
            text, color = f"head {d['turn']:+.2f} TURNED, no neighbour there", (0, 220, 255)
        else:
            text, color = f"head {d['turn']:+.2f} -> NEIGHBOUR {d['held']:.1f}/{seconds_needed:.0f}s", (0, 255, 0)
        x, y = int(p.box[0]) + 6, int(p.box[1]) + int(48 * fs) + 14
        _label(view, text, x, y, fs, color)
    banner = f"people: {len(persons)}   phones: {len(phones)}"
    _label(view, banner, 10, int(30 * fs) + 10, fs, (255, 255, 255))


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
    api = ApiClient(s.api_url, s.api_key)
    pose_model = YOLO(s.pose_model)
    phone_model = YOLO(s.model) if s.enable_phone else None

    rules, cooldowns = [], {}
    if s.enable_phone:
        rules.append(PhoneRule(s))
        cooldowns["phone"] = s.cooldown_seconds
    if s.enable_head_turn:
        rules.append(HeadTurnRule(s))
        cooldowns["looking_at_neighbor"] = s.head_cooldown_seconds

    buffer: deque = deque()                       # (t, small_frame) for the last pre_seconds
    last_alert: dict[tuple, float] = {}           # (behavior, owner) -> time of last alert
    pending: list[PendingClip] = []
    t0, idx = None, 0

    print(f"Running on {s.source} (live={stream.live}), session {s.session_id}. "
          f"Rules: {[r.behavior for r in rules]}. Ctrl+C to stop.")
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

            pose_res = pose_model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                imgsz=s.imgsz, device=s.device, verbose=False,
            )[0]
            persons = extract_persons(pose_res)

            phones = []
            if phone_model is not None:
                phone_res = phone_model.predict(
                    frame, classes=[PHONE], conf=0.25, imgsz=s.imgsz, device=s.device, verbose=False
                )[0]
                phones = extract_phones(phone_res, s.phone_conf)

            for rule in rules:
                for ev in rule.update(t, persons, phones):
                    key = (ev.behavior, ev.owner)
                    if t - last_alert.get(key, -math.inf) < cooldowns[ev.behavior]:
                        continue
                    last_alert[key] = t
                    print(f"[{t - t0:7.1f}s] {ev.behavior.upper()} flagged "
                          f"(student track {ev.owner}, score {ev.conf:.2f}) {ev.details}")
                    try:
                        alert = api.create_alert({
                            "session_id": s.session_id,
                            "track_id": ev.owner if ev.owner != -1 else None,
                            "behavior_type": ev.behavior,
                            "confidence": round(min(1.0, ev.conf), 3),
                            "video_ts_seconds": round(t - t0, 2),
                            "details": ev.details,
                        })
                        pending.append(PendingClip(alert["id"], list(buffer), t + s.post_seconds))
                    except httpx.HTTPError as e:
                        print(f"  failed to send alert: {e}")

            if s.show:
                view = pose_res.plot()
                for conf, (x1, y1, x2, y2) in phones:
                    cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(view, f"phone {conf:.2f}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                head_rule = next((r for r in rules if isinstance(r, HeadTurnRule)), None)
                draw_status(view, persons, phones, head_rule, s.head_turn_seconds)
                cv2.imshow("proctor-detector", resize_to_width(view, 1280))
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
    p = argparse.ArgumentParser(description="Exam hall cheating detector")
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
