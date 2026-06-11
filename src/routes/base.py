from fastapi import APIRouter , FastAPI , Depends
from helpers.config import get_settings

base_router = APIRouter(
    prefix="/api/v1",
    tags=["base"]
)

@base_router.get("/")
async def welcome():
    app_settings = get_settings()

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {"message": f"Welcome to {app_name} version {app_version}!"}