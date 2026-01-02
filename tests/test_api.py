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


def test_predict_with_categorical_features(client):
    """Verify that the endpoint /predict handles categorical features correctly."""
    response = client.post(
        "/predict",
        data={
            "quantity": "1000",
            "isrc": "USRC17607839",
            "continent": "North America",
            "zone": "USA"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 1000
    assert data["isrc"] == "USRC17607839"
    assert data["continent"] == "North America"
    assert data["zone"] == "USA"
    assert data["predicted_revenue"] > 0


def test_predict_with_partial_categorical_features(client):
    """Verify that the endpoint /predict handles partial categorical features correctly."""
    response = client.post(
        "/predict",
        data={
            "quantity": "5000",
            "continent": "Europe"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 5000
    assert data["isrc"] == "UNKNOWN"  # Should use default
    assert data["continent"] == "Europe"
    assert data["zone"] == "UNKNOWN"  # Should use default
    assert data["predicted_revenue"] > 0


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


def test_predict_invalid_continent(client):
    """Verify that the endpoint /predict handles invalid continent gracefully."""
    response = client.post(
        "/predict",
        data={
            "quantity": "1000",
            "continent": "Antarctica"  # Not in training data
        }
    )
    # Should handle gracefully (use UNKNOWN or raise error)
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        data = response.json()
        assert data["predicted_revenue"] > 0


def test_predict_empty_categorical_strings(client):
    """Verify that the endpoint /predict handles empty strings correctly."""
    response = client.post(
        "/predict",
        data={
            "quantity": "1000",
            "isrc": "",
            "continent": "",
            "zone": ""
        }
    )
    assert response.status_code == 200
    data = response.json()
    # Empty strings should be treated as defaults
    assert data["isrc"] == "UNKNOWN"
    assert data["continent"] == "Europe"  # Default
    assert data["zone"] == "UNKNOWN"


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
    assert set(data.keys()) == {"quantity", "isrc", "continent", "zone", "predicted_revenue"}
    
    # Check data types
    assert isinstance(data["quantity"], (int, float))
    assert isinstance(data["isrc"], str)         # Now strings
    assert isinstance(data["continent"], str)    # Now strings
    assert isinstance(data["zone"], str)         # Now strings
    assert isinstance(data["predicted_revenue"], (int, float))
    
    # Check default values are used when not specified
    assert data["isrc"] == "UNKNOWN"
    assert data["continent"] == "Europe"
    assert data["zone"] == "UNKNOWN"


def test_predict_consistency(client):
    """Verify that the same input produces the same output."""
    response1 = client.post(
        "/predict",
        data={
            "quantity": "1000",
            "continent": "Europe"
        }
    )
    response2 = client.post(
        "/predict",
        data={
            "quantity": "1000",
            "continent": "Europe"
        }
    )
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    assert data1["predicted_revenue"] == data2["predicted_revenue"]
