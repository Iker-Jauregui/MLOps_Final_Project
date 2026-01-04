"""
Unit Testing of the track revenue prediction logic
"""
import pytest
import numpy as np
from logic.regressor import predict


def test_predict_single_value():
    """Test predict function with a single quantity value."""
    result = predict(1000)
    assert isinstance(result, float)
    assert result > 0


def test_predict_with_categorical_features():
    """Test predict function with categorical features."""
    result = predict(
        quantity=1000,
        isrc="USRC17607839",
        continent="North America",
        zone="USA"
    )
    assert isinstance(result, float)
    assert result > 0


def test_predict_with_partial_categorical_features():
    """Test predict function with partial categorical features."""
    result = predict(
        quantity=1000,
        continent="Europe"
    )
    assert isinstance(result, float)
    assert result > 0


def test_predict_with_unknown_categorical():
    """Test predict function with unknown categorical value."""
    result = predict(
        quantity=1000,
        isrc="UNKNOWN_ISRC_CODE",  # Not in training data
        continent="Europe"
    )
    # Should handle gracefully and return a prediction
    assert isinstance(result, float)
    assert result > 0


def test_predict_zero():
    """Test predict function with zero quantity."""
    result = predict(0)
    assert result == pytest.approx(0.0, abs=0.1)


def test_predict_large_quantity():
    """Test predict function with large quantity."""
    result = predict(500000)
    assert result > 100


def test_predict_numpy_array():
    """Test predict function with numpy array input."""
    quantities = np.array([1000, 5000, 10000])
    result = predict(quantities)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.all(result > 0)


def test_predict_list_input():
    """Test predict function with list input."""
    quantities = [1000, 2000, 3000]
    result = predict(quantities)
    
    # Check correct number of predictions
    assert len(result) == 3
    # Check all are positive
    assert np.all(result > 0)


def test_predict_float_precision():
    """Test predict function maintains float precision."""
    result = predict(1234)
    assert isinstance(result, float)
    assert result > 0


def test_predict_very_small_quantity():
    """Test predict function with very small quantity."""
    result = predict(1)
    assert result > 0
    assert result < 1  # Should be small


def test_predict_negative_quantity():
    """Test predict function with negative quantity raises ValueError."""
    with pytest.raises(ValueError, match="'quantity' must be non-negative"):
        predict(-1000)


def test_predict_negative_in_array():
    """Test predict function with negative value in array raises ValueError."""
    quantities = np.array([1000, -500, 10000])
    with pytest.raises(ValueError, match="'quantity' must be non-negative"):
        predict(quantities)


def test_predict_negative_in_list():
    """Test predict function with negative value in list raises ValueError."""
    quantities = [1000, -200, 3000]
    with pytest.raises(ValueError, match="'quantity' must be non-negative"):
        predict(quantities)


def test_predict_monotonicity():
    """Test that predictions increase monotonically with quantity."""
    quantities = [100, 1000, 10000, 100000]
    predictions = [predict(q) for q in quantities]
    
    # Check monotonic increase
    for i in range(len(predictions) - 1):
        assert predictions[i] < predictions[i + 1], \
            f"Prediction should increase: {predictions[i]} < {predictions[i+1]}"


def test_predict_default_continent():
    """Test that default continent is Europe."""
    result_with_default = predict(1000)
    result_with_europe = predict(1000, continent="Europe")
    
    # Should be the same prediction
    assert result_with_default == pytest.approx(result_with_europe, rel=1e-6)


def test_predict_empty_string_handling():
    """Test that empty strings are handled as None."""
    result = predict(
        quantity=1000,
        isrc="",
        continent="",
        zone=""
    )
    assert isinstance(result, float)
    assert result > 0


def test_predict_none_handling():
    """Test that None values use defaults."""
    result = predict(
        quantity=1000,
        isrc=None,
        continent=None,
        zone=None
    )
    assert isinstance(result, float)
    assert result > 0

    
def test_predict_with_training_isrc_codes():
    """Test prediction with actual ISRC codes from training data."""
    # Use first few ISRCs from the metadata
    isrc_codes = ["CA-5KR-00-04598", "CA-5KR-00-04599", "DG-A05-22-56057"]
    
    for isrc in isrc_codes:
        result = predict(
            quantity=5000,
            isrc=isrc,
            continent="Europe"
        )
        assert isinstance(result, float)
        assert result > 0


def test_predict_with_all_continents():
    """Test that all continents are properly encoded."""
    continents = ["Africa", "Asia", "Europe", "LATAM", "North America", "Oceania", "Other"]
    
    predictions = []
    for continent in continents:
        result = predict(quantity=10000, continent=continent)
        predictions.append(result)
        assert isinstance(result, float)
        assert result > 0
    
    # Predictions should vary based on continent
    assert len(set(predictions)) > 1  # Not all the same


def test_predict_with_training_zones():
    """Test prediction with actual zone values from training data."""
    zones = ["France", "Germany", "United states", "Canada", "Japan"]
    
    for zone in zones:
        result = predict(
            quantity=10000,
            continent="Europe",
            zone=zone
        )
        assert isinstance(result, float)
        assert result > 0


def test_encode_categorical_coverage():
    """Test the encoding function directly."""
    from logic.regressor import _encode_categorical
    
    # Test with valid values
    assert _encode_categorical("Europe", "continent") > 0
    assert _encode_categorical("CA-5KR-00-04598", "ISRC") > 0
    
    # Test with None
    assert _encode_categorical(None, "continent") == 0
    
    # Test with empty string
    assert _encode_categorical("", "zone") == 0
    
    # Test with unknown value
    assert _encode_categorical("MadeUpValue", "continent") == 0


def test_predict_array_with_different_sizes():
    """Test array predictions with various sizes."""
    sizes = [1, 5, 10, 100]
    
    for size in sizes:
        quantities = np.full(size, 1000.0)
        result = predict(quantities)
        assert len(result) == size
        assert all(r > 0 for r in result)
