from fastapi import FastAPI
from src.routes.base import base_router
from helpers.config import get_settings

settings = get_settings()
app = FastAPI()
app.include_router(base_router)