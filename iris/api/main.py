"""
FastAPI application for Leocherd Iris Energy REST services.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iris.api.routes import assets_router, health_router, jobs_router, uploads_router

app = FastAPI(
    title="Leocherd Iris Energy API",
    description="Universal AI Video-to-3D Rigging and Motion Reconstruction Engine",
    version="0.1.0",
)

# CORS configuration for Web / Three.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes under /api/v1
app.include_router(health_router, prefix="/api/v1")
app.include_router(uploads_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(assets_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "engine": "Leocherd Iris Energy",
        "version": "0.1.0",
        "documentation": "/docs",
        "api_v1": "/api/v1",
    }
