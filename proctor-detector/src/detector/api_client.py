import httpx


class ApiClient:
    def __init__(self, base_url: str, api_key: str):
        self.http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=20)

    def create_alert(self, payload: dict) -> dict:
        r = self.http.post("/alerts", json=payload)
        r.raise_for_status()
        return r.json()

    def upload_clip(self, alert_id: int, path: str) -> None:
        with open(path, "rb") as f:
            r = self.http.post(f"/alerts/{alert_id}/clip", files={"file": (f"alert_{alert_id}.mp4", f, "video/mp4")})
        r.raise_for_status()
