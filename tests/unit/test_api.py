"""
Unit tests for FastAPI REST service endpoints.
"""

from fastapi.testclient import TestClient
import pytest
from iris.api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "Leocherd Iris Energy"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["cpu_cores"] >= 1
    assert data["free_ram_gb"] > 0


def test_project_creation():
    response = client.post("/api/v1/projects", json={"name": "Test Project"})
    assert response.status_code == 200
    data = response.json()
    assert "project_id" in data
    assert data["name"] == "Test Project"
