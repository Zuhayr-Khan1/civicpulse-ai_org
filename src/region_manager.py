"""
Geographic boundary manager for hierarchical region processing
Handles: National, State, District, City levels with parallel execution
Windows PowerShell compatible paths
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box, Polygon
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class RegionBoundary:
    """Represents a geographic region with metadata"""
    
    def __init__(self, name: str, bounds: Tuple[float, float, float, float], 
                 level: str, parent_id: Optional[str] = None):
        """
        Args:
            name: Region name (e.g., "Telangana", "India")
            bounds: (minx, miny, maxx, maxy) in WGS84
            level: "national", "state", "district", "city"
            parent_id: ID of parent region
        """
        self.name = name
        self.bounds = bounds
        self.level = level
        self.parent_id = parent_id
        self.geometry = box(*bounds)
        self.area_km2 = self._calc_area()
    
    def _calc_area(self) -> float:
        """Calculate area in km²"""
        minx, miny, maxx, maxy = self.bounds
        # 1 degree ≈ 111.32 km
        width_km = (maxx - minx) * 111.32
        height_km = (maxy - miny) * 111.32
        return width_km * height_km
    
    def grid_cell_count(self, resolution_km: float = 1.0) -> int:
        """Estimate grid cells at given resolution"""
        return int(self.area_km2 / (resolution_km ** 2))
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'bounds': self.bounds,
            'level': self.level,
            'parent_id': self.parent_id,
            'area_km2': self.area_km2,
            'grid_cells_1km': self.grid_cell_count(1.0)
        }


class ConfigurableBoundaryManager:
    """Manages preset and custom regions with hierarchical support"""
    
    # India-wide bounds (WGS84)
    INDIA_BOUNDS = (68.7, 8.4, 97.5, 35.0)
    
    # Major state boundaries (simplified)
    STATE_BOUNDS = {
        'Telangana': (78.0, 15.5, 81.9, 19.8),
        'Andhra Pradesh': (77.0, 12.6, 84.9, 18.3),
        'Maharashtra': (72.6, 16.5, 80.9, 23.3),
        'Karnataka': (74.0, 11.5, 78.6, 18.6),
        'Tamil Nadu': (76.8, 8.0, 80.3, 13.6),
        'Uttar Pradesh': (77.0, 23.8, 84.8, 30.4),
        'Delhi': (76.76, 28.41, 77.35, 28.88),
        'Punjab': (73.7, 29.5, 76.9, 32.3),
        'Gujarat': (68.1, 20.1, 74.4, 24.5),
        'Haryana': (76.4, 27.0, 77.6, 30.6),
        'Rajasthan': (68.8, 23.0, 78.6, 32.3),
        'West Bengal': (85.8, 21.6, 89.9, 27.2),
        'Kerala': (74.9, 8.3, 77.4, 12.5),
        'Madhya Pradesh': (74.0, 17.8, 82.9, 26.9),
        'Bihar': (82.2, 24.3, 88.3, 27.5),
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize boundary manager"""
        self.regions: Dict[str, RegionBoundary] = {}
        self._load_defaults()
        if config_path:
            self._load_custom(config_path)
    
    def _load_defaults(self):
        """Load preset regions"""
        # National level
        self.add_region(
            RegionBoundary('India', self.INDIA_BOUNDS, 'national')
        )
        
        # State level
        for state_name, bounds in self.STATE_BOUNDS.items():
            self.add_region(
                RegionBoundary(state_name, bounds, 'state', parent_id='India')
            )
    
    def _load_custom(self, config_path: str):
        """Load custom regions from JSON"""
        with open(config_path, 'r') as f:
            custom = json.load(f)
            for region_data in custom['regions']:
                region = RegionBoundary(
                    name=region_data['name'],
                    bounds=tuple(region_data['bounds']),
                    level=region_data['level'],
                    parent_id=region_data.get('parent_id')
                )
                self.add_region(region)
    
    def add_region(self, region: RegionBoundary):
        """Register a region"""
        self.regions[region.name] = region
    
    def get_region(self, name: str) -> Optional[RegionBoundary]:
        """Retrieve region by name"""
        return self.regions.get(name)
    
    def get_regions_by_level(self, level: str) -> List[RegionBoundary]:
        """Get all regions at given level"""
        return [r for r in self.regions.values() if r.level == level]
    
    def get_child_regions(self, parent_name: str) -> List[RegionBoundary]:
        """Get all child regions of a parent"""
        parent_id = parent_name
        return [r for r in self.regions.values() if r.parent_id == parent_id]
    
    def create_hierarchical_grid(self, 
                                top_level: str = 'national',
                                bottom_level: str = 'state',
                                resolution_km: float = 1.0) -> Dict[str, int]:
        """Create hierarchical processing structure"""
        processing_map = {}
        
        for region in self.get_regions_by_level(top_level):
            processing_map[region.name] = region.grid_cell_count(resolution_km)
        
        return processing_map
    
    def to_geojson(self, filename: str = 'regions.geojson'):
        """Export all regions as GeoJSON"""
        features = []
        for region in self.regions.values():
            features.append({
                'type': 'Feature',
                'properties': region.to_dict(),
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [region.bounds[0], region.bounds[1]],
                        [region.bounds[2], region.bounds[1]],
                        [region.bounds[2], region.bounds[3]],
                        [region.bounds[0], region.bounds[3]],
                        [region.bounds[0], region.bounds[1]]
                    ]]
                }
            })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(filename, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} regions to {filename}")


# TEST
if __name__ == '__main__':
    mgr = ConfigurableBoundaryManager()
    india = mgr.get_region('India')
    print(f"India: {india.area_km2:,.0f} km², ~{india.grid_cell_count()//1000}k cells")
    
    states = mgr.get_regions_by_level('state')
    print(f"Configured {len(states)} states")