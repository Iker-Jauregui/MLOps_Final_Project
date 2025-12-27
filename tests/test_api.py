"""
Integration testing with the API
"""
import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


def test_home_endpoint(client):
    """Verify that the endpoint / returns the right message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict(client):
    """Verify that the endpoint /predict performs the revenue prediction correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "1000"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_revenue" in data
    assert "quantity" in data
    assert data["quantity"] == 1000
    assert data["predicted_revenue"] == 2.0


def test_predict_large_quantity(client):
    """Verify that the endpoint /predict handles large quantities correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "500000"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 500000
    assert data["predicted_revenue"] == 1000.0


def test_predict_zero_quantity(client):
    """Verify that the endpoint /predict handles zero quantity correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "0"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 0
    assert data["predicted_revenue"] == 0.0


def test_predict_decimal_quantity(client):
    """Verify that the endpoint /predict handles decimal quantities correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "1500.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 1500.5
    assert data["predicted_revenue"] == pytest.approx(3.001, rel=1e-9)


def test_predict_negative_quantity(client):
    """Verify that the endpoint /predict manages correctly negative quantities."""
    response = client.post(
        "/predict",
        data={"quantity": "-1000"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "'quantity' must be a non-negative value" in data["detail"]


def test_predict_invalid_quantity_string(client):
    """Verify that the endpoint /predict manages correctly invalid string inputs."""
    response = client.post(
        "/predict",
        data={"quantity": "not_a_number"}
    )
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
    data = response.json()
    assert "detail" in data


def test_predict_missing_parameter(client):
    """Verify that the endpoint /predict manages correctly missing parameters."""
    response = client.post("/predict")
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
    data = response.json()
    assert "detail" in data


def test_predict_very_small_quantity(client):
    """Verify that the endpoint /predict handles very small quantities correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 1
    assert data["predicted_revenue"] == pytest.approx(0.002, rel=1e-9)


def test_predict_formula_validation(client):
    """Verify that the prediction formula is correctly applied."""
    test_cases = [
        (1000, 2.0),
        (5000, 10.0),
        (10000, 20.0),
        (25000, 50.0),
    ]
    
    for quantity, expected_revenue in test_cases:
        response = client.post(
            "/predict",
            data={"quantity": str(quantity)}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_revenue"] == pytest.approx(expected_revenue, rel=1e-9)


def test_predict_response_structure(client):
    """Verify that the response has the correct structure."""
    response = client.post(
        "/predict",
        data={"quantity": "1000"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check that all expected keys are present
    assert set(data.keys()) == {"quantity", "predicted_revenue"}
    
    # Check data types
    assert isinstance(data["quantity"], (int, float))
    assert isinstance(data["predicted_revenue"], (int, float))