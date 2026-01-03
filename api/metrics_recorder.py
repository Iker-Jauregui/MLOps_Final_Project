"""
Prometheus metrics for monitoring fraud detection model.
"""
import numpy as np
import threading
import time
from prometheus_client import (
    Counter, Gauge, generate_latest, REGISTRY
)

# Counters
num_of_rmse = Counter(
    "num_of_rmse", 
    "Total number of rmse calculated"
)

# Gauges
obtained_rmse = Gauge(
    'obtained_rmse', 
    'Calculated rmse'
)

class MetricsRecorder:
    def __init__(self):
        sample_rmse = [
            0.46301855, 1.20559233, 0.85118842, 0.83487927, 0.54649135,
            0.52210523, 1.33340921, 1.24418252, 1.75571656, 2.6214048
        ]

        gaussian_noise = 0.04
        results_list = []

        for i in range(len(sample_rmse)):
            if i == 0:
                result = self._generate_range_with_noise(
                    sample_rmse[i] - 0.1, 
                    sample_rmse[i], 
                    gaussian_noise
                )
            else:
                result = self._generate_range_with_noise(
                    sample_rmse[i-1], 
                    sample_rmse[i], 
                    gaussian_noise
                )
            results_list.append(result)

        self.final_array = np.concatenate(results_list)
        self.current_index = 0
        
        # Set initial value immediately
        self.record_rmse(self.final_array[0])
        
        # Start background thread
        self._start_metric_updater()
    
    @staticmethod
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

    def _start_metric_updater(self):
        """Start background thread to update metrics periodically."""
        def update_metrics():
            while True:
                time.sleep(1)  # Wait before updating (aligns with Prometheus scrape interval)
                
                # Move to next value
                self.current_index = (self.current_index + 1) % len(self.final_array)
                self.record_rmse(self.final_array[self.current_index])
        
        thread = threading.Thread(target=update_metrics, daemon=True)
        thread.start()

    @staticmethod
    def record_rmse(rmse: float):
        """Record a single RMSE value."""
        num_of_rmse.inc()
        obtained_rmse.set(rmse)

    @staticmethod
    def get_metrics():
        """
        Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(REGISTRY)

metrics_recorder = MetricsRecorder()

if __name__ == "__main__":
    print("MetricsRecorder initialized. Metrics updating every 10 seconds...")
    # Keep alive for testing
    try:
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopped.")