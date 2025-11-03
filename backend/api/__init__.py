
from fastapi import APIRouter

# Create a main router for all API endpoints
api_router = APIRouter(prefix="/api")

# Import and include sub-routers
from .feature_selection import router as feature_selection_router
from .health import router as health_router
from .data_management import router as data_management_router

api_router.include_router(feature_selection_router, tags=["Feature Selection"])
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(data_management_router, tags=["Data Management"])