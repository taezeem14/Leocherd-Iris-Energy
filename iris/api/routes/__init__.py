"""
API routes package.
"""

from iris.api.routes.assets import router as assets_router
from iris.api.routes.health import router as health_router
from iris.api.routes.jobs import router as jobs_router
from iris.api.routes.uploads import router as uploads_router

__all__ = ["health_router", "uploads_router", "jobs_router", "assets_router"]
