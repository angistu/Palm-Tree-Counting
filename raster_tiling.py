"""
Raster Tiling Utilities for Palm Tree Counting
Provides functions to split rasters into tiles, process them independently,
and merge results with optional overlap handling
"""

import numpy as np
import rasterio
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class RasterTiler:
    """
    Handles tiling, processing, and merging of large rasters
    """
    
    def __init__(self, raster_path: str, tile_size: int = 512, overlap: int = 50):
        """
        Initialize RasterTiler
        
        Args:
            raster_path: Path to the input raster file
            tile_size: Size of each tile in pixels (default: 512x512)
            overlap: Number of pixels to overlap between tiles for seamless processing (default: 50)
        """
        self.raster_path = raster_path
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Open raster and get metadata
        with rasterio.open(raster_path) as src:
            self.width = src.width
            self.height = src.height
            self.count = src.count  # Number of bands
            self.crs = src.crs
            self.transform = src.transform
            self.dtype = src.dtypes[0]
            self.profile = src.profile
    
    def get_tile_windows(self) -> List[Window]:
        """
        Generate a list of windows for all tiles covering the raster
        
        Returns:
            List of rasterio Window objects
        """
        windows = []
        
        # Calculate tile boundaries with overlap
        step = self.tile_size - self.overlap
        
        for row in range(0, self.height, step):
            for col in range(0, self.width, step):
                # Calculate tile bounds
                row_off = min(row, self.height - self.tile_size)
                col_off = min(col, self.width - self.tile_size)
                
                # Create window with bounds checking
                tile_height = min(self.tile_size, self.height - row_off)
                tile_width = min(self.tile_size, self.width - col_off)
                
                window = Window(col_off, row_off, tile_width, tile_height)
                windows.append(window)
        
        return windows
    
    def read_tile(self, window: Window) -> Tuple[np.ndarray, Dict]:
        """
        Read a single tile from the raster
        
        Args:
            window: rasterio Window object
            
        Returns:
            Tuple of (tile_data, tile_metadata)
            - tile_data: numpy array of shape (bands, height, width)
            - tile_metadata: dict with tile location and bounds info
        """
        with rasterio.open(self.raster_path) as src:
            tile_data = src.read(window=window)
            
            # Get window bounds in CRS coordinates
            bounds = src.window_bounds(window)
            
            tile_metadata = {
                'window': window,
                'bounds': bounds,
                'shape': tile_data.shape,
                'row_off': window.row_off,
                'col_off': window.col_off,
                'width': window.width,
                'height': window.height
            }
        
        return tile_data, tile_metadata
    
    def read_all_tiles(self) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Read all tiles from the raster
        
        Returns:
            Tuple of (tiles_list, metadata_list)
        """
        windows = self.get_tile_windows()
        tiles = []
        metadata = []
        
        for i, window in enumerate(windows):
            tile_data, tile_meta = self.read_tile(window)
            tiles.append(tile_data)
            metadata.append(tile_meta)
            
            if (i + 1) % max(1, len(windows) // 10) == 0:
                print(f"  Loaded tile {i + 1}/{len(windows)}")
        
        return tiles, metadata
    
    def xy_to_tile_index(self, x: float, y: float, tiles_metadata: List[Dict]) -> Optional[int]:
        """
        Find which tile contains a given coordinate (x, y)
        
        Args:
            x, y: Coordinates in raster CRS
            tiles_metadata: List of tile metadata
            
        Returns:
            Index of the tile containing (x, y), or None if outside all tiles
        """
        for idx, meta in enumerate(tiles_metadata):
            bounds = meta['bounds']
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                return idx
        return None
    
    def split_detections_by_tile(self, detections_xy: np.ndarray, 
                                  tiles_metadata: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Split detection coordinates by tile
        
        Args:
            detections_xy: Array of shape (n, 2) with (x, y) coordinates
            tiles_metadata: List of tile metadata
            
        Returns:
            Dict mapping tile_index -> array of detections in that tile
        """
        tile_detections = {}
        
        for x, y in detections_xy:
            tile_idx = self.xy_to_tile_index(x, y, tiles_metadata)
            if tile_idx is not None:
                if tile_idx not in tile_detections:
                    tile_detections[tile_idx] = []
                tile_detections[tile_idx].append([x, y])
        
        # Convert lists to numpy arrays
        for key in tile_detections:
            tile_detections[key] = np.array(tile_detections[key])
        
        return tile_detections
    
    def merge_detections(self, tile_detections: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Merge detections from multiple tiles into single array
        
        Args:
            tile_detections: Dict mapping tile_index -> array of detections
            
        Returns:
            Merged array of all detections
        """
        all_detections = []
        for detections in tile_detections.values():
            if len(detections) > 0:
                all_detections.append(detections)
        
        if len(all_detections) == 0:
            return np.array([]).reshape(0, 2)
        
        return np.vstack(all_detections)
    
    def save_tiles_to_disk(self, output_dir: str = "tiles"):
        """
        Save all tiles to individual files (useful for batch processing)
        
        Args:
            output_dir: Directory to save tiles
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        windows = self.get_tile_windows()
        
        for i, window in enumerate(windows):
            tile_data, _ = self.read_tile(window)
            
            # Create output filename
            output_path = os.path.join(output_dir, f"tile_{i:04d}.tif")
            
            # Update profile for single tile
            tile_profile = self.profile.copy()
            tile_profile.update({
                'height': tile_data.shape[1],
                'width': tile_data.shape[2],
                'count': tile_data.shape[0]
            })
            
            # Save tile
            with rasterio.open(output_path, 'w', **tile_profile) as dst:
                dst.write(tile_data)
            
            if (i + 1) % max(1, len(windows) // 10) == 0:
                print(f"  Saved tile {i + 1}/{len(windows)} to {output_dir}")
        
        print(f"All tiles saved to {output_dir}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about tiling configuration
        
        Returns:
            Dictionary with tiling statistics
        """
        windows = self.get_tile_windows()
        
        return {
            'raster_size': (self.width, self.height),
            'raster_area': self.width * self.height,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'num_tiles': len(windows),
            'step_size': self.tile_size - self.overlap,
            'estimated_memory_per_tile_mb': (self.tile_size**2 * self.count * 
                                             np.dtype(self.dtype).itemsize / 1024 / 1024)
        }


def apply_processing_to_tiles(tiler: RasterTiler, 
                               processing_func,
                               remove_overlap: bool = True) -> List:
    """
    Process each tile independently using a provided function
    
    Args:
        tiler: RasterTiler instance
        processing_func: Function that takes (tile_data, metadata) and returns results
        remove_overlap: If True, remove overlap from processing results
        
    Returns:
        List of results from each tile
    """
    windows = tiler.get_tile_windows()
    all_results = []
    
    print(f"Processing {len(windows)} tiles...")
    
    for i, window in enumerate(windows):
        tile_data, tile_meta = tiler.read_tile(window)
        
        # Apply processing function
        result = processing_func(tile_data, tile_meta)
        all_results.append(result)
        
        if (i + 1) % max(1, len(windows) // 10) == 0:
            print(f"  Processed tile {i + 1}/{len(windows)}")
    
    return all_results


# Example usage function
def example_tiling_workflow(raster_path: str, tile_size: int = 512, overlap: int = 50):
    """
    Example workflow for raster tiling
    """
    print("=" * 60)
    print("RASTER TILING WORKFLOW")
    print("=" * 60)
    
    # Initialize tiler
    print(f"\nInitializing tiler for {raster_path}...")
    tiler = RasterTiler(raster_path, tile_size=tile_size, overlap=overlap)
    
    # Print statistics
    stats = tiler.get_statistics()
    print(f"\nTiling Statistics:")
    print(f"  Raster size: {stats['raster_size'][0]}x{stats['raster_size'][1]}")
    print(f"  Tile size: {stats['tile_size']}x{stats['tile_size']}")
    print(f"  Overlap: {stats['overlap']} pixels")
    print(f"  Number of tiles: {stats['num_tiles']}")
    print(f"  Est. memory per tile: {stats['estimated_memory_per_tile_mb']:.1f} MB")
    
    # Get tiles
    print(f"\nLoading tiles...")
    tiles, metadata = tiler.read_all_tiles()
    print(f"Successfully loaded {len(tiles)} tiles")
    
    return tiler, tiles, metadata


if __name__ == '__main__':
    # Example usage
    raster_file = 'Ortho/example.tif'
    if os.path.exists(raster_file):
        tiler, tiles, metadata = example_tiling_workflow(raster_file)
