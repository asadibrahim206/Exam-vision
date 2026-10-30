import os

# Must be set before OpenCV opens a stream: TCP is far more reliable than UDP for RTSP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import threading
import time

import cv2


def is_live(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


class VideoStream:
    """Yields (timestamp_seconds, frame).

    Live sources (webcam / RTSP): a background thread always reads the newest frame, so
    inference never falls behind into stale video, and it reconnects if the camera drops.
    Files: read sequentially, no frames skipped, timestamps come from the file's fps.
    """

    def __init__(self, source: str):
        self.live = is_live(source)
        self.source = int(source) if source.isdigit() else source
        self.fps = 25.0
        self._idx = 0
        self._seq = 0
        self._frame = None
        self._cond = threading.Condition()
        self._stop = False
        self._cap = self._open()
        if self._cap is None:
            raise SystemExit(f"Cannot open video source: {source}")
        if self.live:
            threading.Thread(target=self._reader, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and 1 < fps < 120:
            self.fps = fps
        return cap

    def _reader(self):
        cap, attempt = self._cap, 0
        while not self._stop:
            if cap is None:
                wait = min(2 ** attempt, 30)
                print(f"Camera disconnected. Reconnecting in {wait}s...")
                time.sleep(wait)
                cap = self._open()
                attempt = attempt + 1 if cap is None else 0
                continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                continue
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()
        if cap is not None:
            cap.release()

    def read(self):
        """Return (t, frame), or None when the stream has ended / been stopped."""
        if not self.live:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._idx += 1
            return self._idx / self.fps, frame
        with self._cond:
            last = self._seq
            while self._seq == last and not self._stop:
                self._cond.wait(timeout=5)
            if self._stop:
                return None
            return time.monotonic(), self._frame

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        if not self.live and self._cap is not None:
            self._cap.release()
