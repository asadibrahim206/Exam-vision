from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://proctor:proctor@localhost:5432/proctor"
    detector_api_key: str = "change-me"
    clips_dir: str = "clips"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
