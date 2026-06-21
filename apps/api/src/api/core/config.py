from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str
    google_api_key: str
    groq_api_key: str

config = Config()