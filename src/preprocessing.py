"""
Region-aware preprocessing for heterogeneous data quality
Handles: Urban vs rural quality differences, sparse vs dense coverage
Windows PowerShell compatible
"""

import numpy as np
import rasterio
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import warnings


class RegionAwarePreprocessor:
    """
    Analyzes and handles data quality variations across regions
    Urban areas: high quality, dense data
    Rural areas: sparse, interpolation needed
    """
    
    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold
        self.quality_mask = None
        self.confidence_scores = None
    
    def calculate_quality_score(self, data: np.ndarray, 
                               region_type: str = 'mixed') -> np.ndarray:
        """Calculate quality score for each grid cell"""
        height, width = data.shape
        quality = np.zeros_like(data, dtype=np.float32)
        
        # Score 1: Non-zero cells
        nonzero = (data > 0).astype(np.float32)
        quality += 0.2 * nonzero
        
        # Score 2: Spatial consistency
        for i in range(1, height-1):
            for j in range(1, width-1):
                neighbors = data[i-1:i+2, j-1:j+2].flatten()
                valid_neighbors = neighbors[neighbors > 0]
                
                if len(valid_neighbors) > 0:
                    consistency = 1.0 - np.std(valid_neighbors) / (np.mean(valid_neighbors) + 1e-6)
                    quality[i, j] += 0.3 * max(0, consistency)
        
        # Score 3: Region-specific adjustments
        if region_type == 'urban':
            density = np.sum(data > 0) / data.size
            quality *= (1.0 + 0.5 * density)
        elif region_type == 'rural':
            quality = np.minimum(quality * 1.2, 1.0)
        
        quality = np.minimum(quality / (np.max(quality) + 1e-6), 1.0)
        return quality
    
    def identify_low_quality_regions(self, data: np.ndarray, 
                                    threshold: float = 0.3) -> np.ndarray:
        """Identify cells requiring interpolation"""
        quality = self.calculate_quality_score(data)
        return quality < threshold
    
    def adaptive_interpolation(self, data: np.ndarray, 
                              quality_mask: np.ndarray) -> np.ndarray:
        """Adaptive interpolation using KNN"""
        data_filled = data.copy()
        height, width = data.shape
        
        valid_coords = np.argwhere(~quality_mask)
        valid_values = data[~quality_mask]
        
        if len(valid_coords) == 0:
            warnings.warn("No high-quality cells found for interpolation")
            return data_filled
        
        tree = cKDTree(valid_coords)
        missing_coords = np.argwhere(quality_mask)
        
        if len(missing_coords) > 0:
            distances, indices = tree.query(missing_coords, k=5)
            
            for idx, (coord, dist_list, idx_list) in enumerate(
                zip(missing_coords, distances, indices)
            ):
                weights = 1.0 / (dist_list + 1e-6)
                weights /= weights.sum()
                interpolated_value = np.sum(valid_values[idx_list] * weights)
                data_filled[coord[0], coord[1]] = interpolated_value
        
        return data_filled


# TEST
if __name__ == '__main__':
    preprocessor = RegionAwarePreprocessor()
    print("✓ Preprocessor loaded")