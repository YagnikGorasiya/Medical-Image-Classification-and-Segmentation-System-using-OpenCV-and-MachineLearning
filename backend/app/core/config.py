from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Medical CV System"
    MODELS_DIR: str = "models"

settings = Settings()
