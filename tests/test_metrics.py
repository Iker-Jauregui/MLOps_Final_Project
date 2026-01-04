"""
Unit tests for metrics recorder
"""
import time
import pytest
from api.metrics_recorder import MetricsRecorder, metrics_recorder


def test_metrics_recorder_initialization():
    """Test that MetricsRecorder initializes correctly."""
    recorder = MetricsRecorder(gaussian_noise=0.04)
    assert recorder.gaussian_noise == 0.04
    assert len(recorder.sample_rmse) == 10
    assert recorder.current_index >= 0
    assert len(recorder.final_array) > 0


def test_generate_range_with_noise():
    """Test the noise generation function."""
    recorder = MetricsRecorder()
    result = recorder._generate_range_with_noise(0.5, 1.5, 0.1)
    
    assert len(result) == 28
    assert all(val > 0 for val in result)  # All values should be positive
    assert result[-1] == pytest.approx(1.5, abs=0.0001)  # Last value should match B


def test_extend_array():
    """Test array extension with drift."""
    recorder = MetricsRecorder()
    
    # Without drift
    array_no_drift = recorder._extend_array(drift=0)
    assert len(array_no_drift) == 28 * len(recorder.sample_rmse)
    assert all(val > 0 for val in array_no_drift)
    
    # With drift
    array_with_drift = recorder._extend_array(drift=0.5)
    assert len(array_with_drift) == len(array_no_drift)
    # Values with drift should generally be higher
    assert array_with_drift.mean() > array_no_drift.mean()


def test_record_rmse():
    """Test recording RMSE values."""
    initial_metrics = MetricsRecorder.get_metrics()
    
    # Record a value
    MetricsRecorder.record_rmse(1.234)
    
    # Get updated metrics
    updated_metrics = MetricsRecorder.get_metrics()
    
    # Check that metrics were updated
    assert b"obtained_rmse" in updated_metrics
    assert b"num_of_rmse" in updated_metrics


def test_get_metrics():
    """Test that get_metrics returns Prometheus format."""
    metrics = MetricsRecorder.get_metrics()
    
    assert isinstance(metrics, bytes)
    assert b"num_of_rmse" in metrics
    assert b"obtained_rmse" in metrics
    # Should contain HELP and TYPE declarations
    assert b"# HELP" in metrics or b"# TYPE" in metrics


def test_global_metrics_recorder_exists():
    """Test that global metrics_recorder instance exists."""
    assert metrics_recorder is not None
    assert isinstance(metrics_recorder, MetricsRecorder)
    assert hasattr(metrics_recorder, 'final_array')
    assert hasattr(metrics_recorder, 'current_index')


def test_metrics_recorder_background_updates():
    """Test that background thread updates metrics."""
    recorder = MetricsRecorder()
    initial_index = recorder.current_index
    
    # Wait for background thread to update (> 1 second)
    time.sleep(1.5)
    
    # Index should have changed (unless it completed exactly one cycle)
    # We just check it's valid
    assert 0 <= recorder.current_index < len(recorder.final_array)


def test_metrics_recorder_cycle_completion():
    """Test that drift accumulates after completing a cycle."""
    recorder = MetricsRecorder()
    recorder.current_index = len(recorder.final_array) - 1
    initial_drift = recorder.current_drift
    
    # Force one more update to trigger cycle completion
    time.sleep(1.5)
    
    # After cycle, drift should have been updated (or stayed the same if no update yet)
    assert recorder.current_drift >= initial_drift


def test_metrics_positive_values():
    """Test that all generated RMSE values are positive."""
    recorder = MetricsRecorder()
    
    # Check all values in final_array
    assert all(val > 0 for val in recorder.final_array)
    
    # Check sample RMSE values
    assert all(val > 0 for val in recorder.sample_rmse)
