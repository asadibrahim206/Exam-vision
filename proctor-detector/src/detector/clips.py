import cv2


def resize_to_width(frame, width: int):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def write_clip(frames: list, fps: float, path: str) -> bool:
    """Write frames to an mp4. Tries H.264 first (plays in browsers), falls back to mp4v."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            return True
        writer.release()
    return False
