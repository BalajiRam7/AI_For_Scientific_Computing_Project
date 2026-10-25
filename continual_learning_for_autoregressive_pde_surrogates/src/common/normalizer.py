from typing import Dict
import numpy as np


class Normalizer:
    def __init__(self, stats: Dict[str, Dict[str, float]]):
        self.stats = stats

    def normalize(self, data: np.ndarray, key: str):
        """
        Normalizes the data for a given key using the min-max scaling method.
        """
        if key not in self.stats:
            raise ValueError(f"No normalization stats found for key: {key}")

        stat = self.stats[key]
        min_val, max_val = stat["min"], stat["max"]

        if max_val == min_val:
            if isinstance(data, np.ndarray):
                return np.zeros_like(data, dtype=np.float32)
            else:
                return 0.0

        return (data - min_val) / (max_val - min_val)

    def denormalize(self, data: np.ndarray, key: str):
        """
        De-normalizes the data for a given key using the min-max scaling method.
        """
        if key not in self.stats:
            raise ValueError(f"No normalization stats found for key: {key}")

        stat = self.stats[key]
        min_val, max_val = stat["min"], stat["max"]

        return data * (max_val - min_val) + min_val
