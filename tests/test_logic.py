"""
Unit Testing of the track revenue prediction logic
"""
import pytest
import numpy as np
from logic.regressor import predict


def test_predict_single_value():
    """Test predict function with a single quantity value."""
    result = predict(1000)
    assert result == 2.0


def test_predict_zero():
    """Test predict function with zero quantity."""
    result = predict(0)
    assert result == 0.0


def test_predict_large_quantity():
    """Test predict function with large quantity."""
    result = predict(500000)
    assert result == 1000.0


def test_predict_numpy_array():
    """Test predict function with numpy array input."""
    quantities = np.array([1000, 5000, 10000])
    result = predict(quantities)
    expected = np.array([2.0, 10.0, 20.0])
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, expected)


def test_predict_list_input():
    """Test predict function with list input (numpy broadcasting)."""
    quantities = [1000, 2000, 3000]
    result = predict(quantities)
    expected = [2.0, 4.0, 6.0]
    
    # Convert to numpy arrays for comparison
    np.testing.assert_array_almost_equal(result, expected)


def test_predict_float_precision():
    """Test predict function maintains float precision."""
    result = predict(1234)
    assert isinstance(result, float)
    assert result == pytest.approx(2.468, rel=1e-9)


def test_predict_very_small_quantity():
    """Test predict function with very small quantity."""
    result = predict(1)
    assert result == pytest.approx(0.002, rel=1e-9)


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
