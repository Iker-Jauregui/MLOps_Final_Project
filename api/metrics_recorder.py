"""
Prometheus metrics for monitoring fraud detection model.
"""
import numpy as np

from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, REGISTRY
)

# Counters
predictions_made = Counter(
    "predictions_made", 
    "Total number of predictions made"
)

# Gauges
estimated_revenue = Gauge(
    'estimated_revenue', 
    'Estimated revenue inferred'
)

def _generate_range_with_noise(A, B, C):
    """
    Generates a linear range of 28 points between A and B with Gaussian noise.
    All values are guaranteed to be positive.
    
    Parameters:
    -----------
    A : float
        Lower bound of the range (must be positive)
    B : float
        Upper bound of the range (must be positive)
    C : float
        Standard deviation of the Gaussian noise (not applied to the last point)
    
    Returns:
    --------
    numpy.array : Array of 28 positive values with Gaussian noise (except the last one)
    """
    # Create linear range of 28 points
    range_array = np.linspace(A, B, 28)
    
    # Generate Gaussian noise for the first 27 points
    noise = np.random.normal(0, C, 27)
    
    # Add noise to all points except the last one
    range_array[:-1] += noise
    
    # Ensure all values are positive (clip at a small positive value)
    range_array = np.maximum(range_array, 0.0001)
    
    return range_array

def record_prediction(prediction: int):
    predictions_made.inc()
    estimated_revenue.set(prediction)

def get_metrics():
    """
    Get current metrics in Prometheus format.

    Returns:
        Metrics in Prometheus exposition format
    """
    return generate_latest(REGISTRY)

if __name__ == "__main__":
    start_http_server(8000)

    predictions = [
        0.46301855, 1.20559233, 0.85118842, 0.83487927, 0.54649135, \
        0.52210523, 1.33340921, 1.24418252, 1.75571656, 2.6214048
    ]

    noise = 0.04
    results_list = []

    for i in range(len(predictions)):
        if i == 0:
            result = _generate_range_with_noise(
                predictions[i] - 0.1, 
                predictions[i], 
                noise
            )

        else:
            result = _generate_range_with_noise(
                predictions[i-1], 
                predictions[i], 
                noise
            )
        results_list.append(result)

    final_array = np.concatenate(results_list)

    for i, prediction in enumerate(final_array):
        record_prediction(prediction)
