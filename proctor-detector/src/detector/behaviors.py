"""Rule-based behaviour detectors. Each rule gets per-frame observations and returns Events."""
import math
from collections import deque
from dataclasses import dataclass, field

NOSE, L_EYE, R_EYE = 0, 1, 2


@dataclass
class Person:
    id: int | None
    box: list                 # x1, y1, x2, y2
    kpts: object = None       # (17, 2) array or None
    kconf: object = None      # (17,) array or None


@dataclass
class Event:
    behavior: str
    owner: int                # tracker id, -1 if unknown
    conf: float
    details: dict = field(default_factory=dict)


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def assign_owner(phone_box, persons):
    """Tracker id of the person this phone most likely belongs to (-1 if unknown)."""
    cx, cy = _center(phone_box)
    best_id, best_d = -1, math.inf
    for p in persons:
        pcx, pcy = _center(p.box)
        inside = p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3]
        d = math.hypot(cx - pcx, cy - pcy)
        if not inside and d > (p.box[3] - p.box[1]):
            continue
        if d < best_d:
            best_id, best_d = (p.id if p.id is not None else -1), d
    return best_id


class PhoneRule:
    behavior = "phone"

    def __init__(self, s):
        self.window_frames, self.confirm_frames = s.window_frames, s.confirm_frames
        self.hits: dict[int, deque] = {}

    def update(self, t, persons, phones):
        seen: dict[int, tuple[float, list]] = {}
        for conf, box in phones:
            owner = assign_owner(box, persons)
            if owner not in seen or conf > seen[owner][0]:
                seen[owner] = (conf, box)
        for owner in seen:
            self.hits.setdefault(owner, deque(maxlen=self.window_frames))
        events = []
        for owner, window in list(self.hits.items()):
            window.append(owner in seen)
            if len(window) == window.maxlen and not any(window):
                del self.hits[owner]
                continue
            if sum(window) >= self.confirm_frames:
                conf, box = seen.get(owner, (0.0, []))
                events.append(Event(self.behavior, owner, conf,
                                    {"bbox": [round(v, 1) for v in box], "frames_hit": int(sum(window))}))
        return events


def head_turn(p: Person):
    """Signed head-turn estimate in image space (+ = nose points image-right), or None.

    Uses nose position relative to the eye midline, normalised by eye distance, so it works
    at any scale. ~0 when facing the camera, ~0.8*tan(yaw) when turned. If only one eye is
    visible the head is turned a lot, so we return +-1.
    """
    if p.kpts is None or p.kconf is None:
        return None
    nose, le, re = p.kpts[NOSE], p.kpts[L_EYE], p.kpts[R_EYE]
    cn, cl, cr = p.kconf[NOSE], p.kconf[L_EYE], p.kconf[R_EYE]
    if cn < 0.5:
        return None
    if cl >= 0.5 and cr >= 0.5:
        eye_dist = abs(le[0] - re[0])
        if eye_dist < 6:  # too small to trust
            return None
        return (nose[0] - (le[0] + re[0]) / 2) / eye_dist
    visible = le if (cl >= 0.5 and cr < 0.3) else re if (cr >= 0.5 and cl < 0.3) else None
    if visible is None:
        return None
    return 1.0 if nose[0] > visible[0] else -1.0 if nose[0] < visible[0] else None


def neighbor_in_direction(p: Person, persons, direction: int) -> bool:
    """Is another student sitting to the image-left (-1) / image-right (+1) of p, close and in the same row?"""
    pcx, pcy = _center(p.box)
    width, height = p.box[2] - p.box[0], p.box[3] - p.box[1]
    for q in persons:
        if q is p:
            continue
        qcx, qcy = _center(q.box)
        dx, dy = qcx - pcx, abs(qcy - pcy)
        if dx * direction > 0 and abs(dx) <= 3 * width and dy <= 0.75 * height:
            return True
    return False


class HeadTurnRule:
    behavior = "looking_at_neighbor"
    GRACE = 0.7   # seconds the turn may be interrupted (blink, tracker jitter) before we reset

    def __init__(self, s):
        self.ratio, self.seconds = s.head_turn_ratio, s.head_turn_seconds
        self.start: dict[int, float] = {}
        self.direction: dict[int, int] = {}
        self.last_true: dict[int, float] = {}
        self.debug: dict[int, dict] = {}   # live per-student state, used by the --show overlay

    def update(self, t, persons, phones=None):
        events = []
        self.debug = {}
        for p in persons:
            if p.id is None:
                continue
            turn = head_turn(p)
            facing = None
            nbr = None
            if turn is not None and abs(turn) >= self.ratio:
                d = 1 if turn > 0 else -1
                nbr = neighbor_in_direction(p, persons, d)
                if nbr:
                    facing = d
            self.debug[p.id] = {
                "turn": turn,
                "nbr": nbr,
                "held": (t - self.start[p.id]) if (facing is not None and p.id in self.start) else 0.0,
            }
            if facing is not None:
                if p.id not in self.start or self.direction[p.id] != facing:
                    self.start[p.id], self.direction[p.id] = t, facing
                self.last_true[p.id] = t
                held = t - self.start[p.id]
                self.debug[p.id]["held"] = held
                if held >= self.seconds:
                    events.append(Event(self.behavior, p.id, min(1.0, abs(turn)),
                                        {"head_turn": round(turn, 2), "held_seconds": round(held, 1),
                                         "toward": "image-right" if facing > 0 else "image-left"}))
            elif p.id in self.start and t - self.last_true.get(p.id, t) > self.GRACE:
                self.start.pop(p.id, None)
                self.direction.pop(p.id, None)
        for pid in [k for k, v in self.last_true.items() if t - v > 60]:  # forget students who left
            self.last_true.pop(pid, None)
            self.start.pop(pid, None)
            self.direction.pop(pid, None)
        return events
