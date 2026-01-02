"""
Integration testing with the API
"""
import pytest
from fastapi.testclient import TestClient
from api.api import app


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
    assert data["predicted_revenue"] > 0
    assert data["predicted_revenue"] < 100  # Sanity check


def test_predict_large_quantity(client):
    """Verify that the endpoint /predict handles large quantities correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "500000"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 500000
    assert data["predicted_revenue"] > 100
    assert data["predicted_revenue"] < 10000


def test_predict_zero_quantity(client):
    """Verify that the endpoint /predict handles zero quantity correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "0"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 0
    assert data["predicted_revenue"] == pytest.approx(0.0, abs=0.1)


def test_predict_decimal_quantity(client):
    """Verify that the endpoint /predict handles decimal quantities correctly."""
    response = client.post(
        "/predict",
        data={"quantity": "1500.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 1500.5
    assert data["predicted_revenue"] > 0


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
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_missing_parameter(client):
    """Verify that the endpoint /predict manages correctly missing parameters."""
    response = client.post("/predict")
    assert response.status_code == 422
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
    assert 0 < data["predicted_revenue"] < 1


def test_predict_monotonicity(client):
    """Verify that predictions increase with quantity (monotonicity test)."""
    quantities = [1000, 5000, 10000, 25000]
    predictions = []
    
    for quantity in quantities:
        response = client.post("/predict", data={"quantity": str(quantity)})
        assert response.status_code == 200
        predictions.append(response.json()["predicted_revenue"])
    
    # Check that predictions are monotonically increasing
    for i in range(len(predictions) - 1):
        assert predictions[i] < predictions[i + 1], \
            f"Prediction for {quantities[i]} should be less than {quantities[i+1]}"


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