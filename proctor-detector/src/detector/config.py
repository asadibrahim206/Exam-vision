from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # backend
    api_url: str = "http://127.0.0.1:8000"
    api_key: str = "change-me"
    session_id: int = 1

    # video + model
    source: str = "0"
    model: str = "yolo11m.pt"              # phone detector
    pose_model: str = "yolo11s-pose.pt"    # people + face keypoints (head-turn)
    enable_phone: bool = True
    enable_head_turn: bool = True
    device: str | None = None        # "cpu", "0" (first GPU), ...
    imgsz: int = 960                 # inference size; larger helps tiny phones, costs speed
    frame_skip: int = 1              # run the model on every Nth frame
    show: bool = False

    # phone logic
    phone_conf: float = 0.35         # min confidence for a single detection
    window_frames: int = 15          # look at the last N processed frames...
    confirm_frames: int = 8          # ...and flag if the phone is seen in at least M of them
    cooldown_seconds: float = 30.0   # don't re-flag the same student within this time

    # head-turn (looking at neighbor)
    head_turn_ratio: float = 0.6     # ~0.8*tan(yaw): 0.6 is roughly a 35-40 degree turn
    head_turn_seconds: float = 3.0   # must be held this long
    head_cooldown_seconds: float = 30.0

    # evidence clip
    pre_seconds: float = 4.0
    post_seconds: float = 3.0
    clip_width: int = 960

    model_config = SettingsConfigDict(env_file=".env")
