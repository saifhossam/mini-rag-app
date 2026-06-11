from pydantic import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    class Config:
        env_file = ".env"   

    def get_settings(self):
        return {
            "APP_NAME": self.APP_NAME,
            "APP_VERSION": self.APP_VERSION
        }