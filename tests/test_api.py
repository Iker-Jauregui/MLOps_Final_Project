"""
Integration testing with the API
"""
import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image in memory for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_home_endpoint(client):
    """Verify that the endpoint / returns the right message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict(client, sample_image_bytes):
    """Verify that the endpoint /predict performs the class prediction correctly."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"class_names": "cat,dog,bird"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["cat", "dog", "bird"]


def test_predict_invalid_file(client):
    """Verify that the endpoint /predict manages correctly invalid files."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_resize(client, sample_image_bytes):
    """Verify that the endpoint /resize performs the image resize correctly."""
    response = client.post(
        "/resize",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"width": "32", "height": "32"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "resized_dimensions" in data
    assert data["resized_dimensions"] == [32, 32]


def test_resize_invalid_width(client, sample_image_bytes):
    """Verify that the endpoint /resize manages correctly invalid widths."""
    response = client.post(
        "/resize",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"width": "0", "height": "32"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "'width' must be a positive value" in data["detail"]


def test_resize_invalid_height(client, sample_image_bytes):
    """Verify that the endpoint /resize manages correctly invalid heights."""
    response = client.post(
        "/resize",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"width": "32", "height": "0"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "'height' must be a positive value" in data["detail"]


def test_resize_invalid_parameters(client):
    """Verify that the endpoint /resize manages correctly missing parameters."""
    response = client.post(
        "/resize",
        data={"width": "32", "height": "32"}
    )
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
    data = response.json()
    assert "detail" in data