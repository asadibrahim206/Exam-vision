"""Save frames from the real exam-hall camera so you can label them and fine-tune the model."""
import argparse
import time
from pathlib import Path

import cv2

from .config import Settings
from .stream import VideoStream


def main() -> None:
    p = argparse.ArgumentParser(description="Capture training frames from a camera/video")
    p.add_argument("--source", help="rtsp url / video file / webcam index (defaults to SOURCE in .env)")
    p.add_argument("--every", type=float, default=5.0, help="seconds between saved frames")
    p.add_argument("--out", default="dataset/images")
    p.add_argument("--max", type=int, default=2000)
    args = p.parse_args()

    source = args.source or Settings().source
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = VideoStream(source)
    last, n = -1e9, 0
    print(f"Saving a full-resolution frame every {args.every}s to {out}/  (Ctrl+C to stop)")
    try:
        while n < args.max:
            item = stream.read()
            if item is None:
                break
            t, frame = item
            if t - last >= args.every:
                last = t
                n += 1
                cv2.imwrite(str(out / f"frame_{int(time.time())}_{n:05d}.jpg"), frame)
                if n % 20 == 0:
                    print(f"{n} frames saved")
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
    print(f"Done. {n} frames in {out}/")


if __name__ == "__main__":
    main()
