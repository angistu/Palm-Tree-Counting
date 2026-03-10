#!/usr/bin/env python3
"""
Palm Tree Counting Analysis Script with Raster Tiling
Improved version that processes tiles independently for better memory efficiency
and accuracy
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from skimage.feature import match_template
import numpy as np
from PIL import Image
from sklearn.cluster import Birch
import os
from raster_tiling import RasterTiler

def extract_template_band(greenBand, row, col, ratio=25, n_rotation=12):
    """
    Extract and create template images with rotations
    
    Returns:
        List of template images
    """
    imageList = []
    
    # Bounds check and extract original band
    r_start = max(0, row - ratio)
    r_end = min(greenBand.shape[0], row + ratio)
    c_start = max(0, col - ratio)
    c_end = min(greenBand.shape[1], col + ratio)
    
    template_original = greenBand[r_start:r_end, c_start:c_end]
    if template_original.size > 0:
        imageList.append(template_original)
    
    # Extract larger region for rotations
    r_start_rot = max(0, row - 2*ratio)
    r_end_rot = min(greenBand.shape[0], row + 2*ratio)
    c_start_rot = max(0, col - 2*ratio)
    c_end_rot = min(greenBand.shape[1], col + 2*ratio)
    
    templateBandToRotate = greenBand[r_start_rot:r_end_rot, c_start_rot:c_end_rot]
    
    if templateBandToRotate.size > 0:
        rotationList = [i*30 for i in range(1, n_rotation)]
        for rotation in rotationList:
            rotatedRaw = Image.fromarray(templateBandToRotate.astype('uint8'))
            rotatedImage = rotatedRaw.rotate(rotation)
            rotated_array = np.asarray(rotatedImage)
            
            # Crop to original size
            if rotated_array.shape[0] > 2*ratio and rotated_array.shape[1] > 2*ratio:
                cropped = rotated_array[ratio:-ratio, ratio:-ratio]
                if cropped.size > 0:
                    imageList.append(cropped)
    
    return imageList


def process_tile_matching(tiles, tile_metadata, palmRaster, templateBandList):
    """
    Process template matching on specified tiles
    
    Args:
        tiles: List of tile arrays
        tile_metadata: List of tile metadata dicts
        palmRaster: Open rasterio dataset
        templateBandList: List of template images
        
    Returns:
        List of matched coordinates (x, y)
    """
    matchXYList = []
    
    for tile_idx, (tile_data, tile_meta) in enumerate(zip(tiles, tile_metadata)):
        print(f"  Processing tile {tile_idx + 1}/{len(tiles)}...")
        
        # Extract green band from tile (assuming band 2 is green, 0-indexed)
        if tile_data.shape[0] >= 2:
            greenBand_tile = tile_data[1]  # Green band
        else:
            continue
        
        # Template matching for this tile
        skipped = 0
        for template_idx, templateband in enumerate(templateBandList):
            if len(templateband.shape) != 2 or templateband.shape[0] < 3 or templateband.shape[1] < 3:
                skipped += 1
                continue
            
            try:
                matchTemplate = match_template(greenBand_tile, templateband, pad_input=True)
                matchTemplateFiltered = np.where(matchTemplate > np.quantile(matchTemplate, 0.9996))
                
                for pixel_row, pixel_col in zip(matchTemplateFiltered[0], matchTemplateFiltered[1]):
                    # Convert pixel coordinates to tile-relative then to full raster coords
                    full_row = tile_meta['row_off'] + pixel_row
                    full_col = tile_meta['col_off'] + pixel_col
                    
                    x, y = palmRaster.xy(full_row, full_col)
                    matchXYList.append([x, y])
            except Exception as e:
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"    Skipped {skipped} invalid templates")
    
    return matchXYList


def main():
    print("=" * 60)
    print("PALM TREE COUNTING WITH RASTER TILING")
    print("=" * 60)
    
    # ===== Step 1: Load Input Data =====
    print("\n[1/8] Loading input shapefile...")
    pointData = gpd.read_file('SHP/sample_point.shp')
    print(f'  CRS of Point Data: {pointData.crs}')
    print(f'  Number of Points: {len(pointData)}')
    
    print("\n[2/8] Loading orthophoto raster...")
    palmRaster = rasterio.open('Ortho/example.tif')
    print(f'  CRS of Raster Data: {palmRaster.crs}')
    print(f'  Raster Size: {palmRaster.width}x{palmRaster.height}')
    print(f'  Number of Bands: {palmRaster.count}')
    
    # ===== Step 2: Initialize Raster Tiler =====
    print("\n[3/8] Initializing raster tiler...")
    tiler = RasterTiler(
        'Ortho/example.tif',
        tile_size=512,      # 512x512 pixel tiles
        overlap=50          # 50 pixel overlap for seamless processing
    )
    
    stats = tiler.get_statistics()
    print(f"  Tiling Configuration:")
    print(f"    Raster size: {stats['raster_size'][0]}x{stats['raster_size'][1]}")
    print(f"    Number of tiles: {stats['num_tiles']}")
    print(f"    Est. memory per tile: {stats['estimated_memory_per_tile_mb']:.1f} MB")
    
    # ===== Step 3: Create Preview Visualization =====
    print("\n[4/8] Creating preview visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    pointData.plot(ax=ax, color='red', markersize=10, label='Sample Points')
    show(palmRaster, ax=ax)
    ax.legend()
    ax.set_title('Input Data: Points and Orthophoto')
    plt.savefig('Output/preview.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: Output/preview.png")
    
    # ===== Step 4: Extract Point Coordinates =====
    print("\n[5/8] Extracting point coordinates...")
    surveyRowCol = []
    
    geom_type = pointData.geom_type.unique()[0]
    if geom_type == "MultiPoint":
        pointData = pointData.explode(index_parts=False)
    
    for index, values in pointData.iterrows():
        geom = values['geometry']
        if hasattr(geom, 'coords'):
            coords = list(geom.coords)[0]
        else:
            coords = (geom.x, geom.y)
        
        x, y = coords
        row, col = palmRaster.index(x, y)
        surveyRowCol.append([row, col])
    
    print(f"  Total points: {len(surveyRowCol)}")
    
    # ===== Step 5: Load All Tiles =====
    print("\n[6/8] Loading all raster tiles...")
    tiles, tile_metadata = tiler.read_all_tiles()
    print(f"  Loaded {len(tiles)} tiles")
    
    # ===== Step 6: Create Templates from Point Data =====
    print("\n[7/8] Creating template images with rotations...")
    ratio = 25
    n_rotation = 12
    templateBandList = []
    
    # Read full green band for template extraction
    greenBand = palmRaster.read(2)  # Green band (0-indexed)
    
    for indeks, rowCol in enumerate(surveyRowCol):
        row = rowCol[0]
        col = rowCol[1]
        
        imageList = extract_template_band(greenBand, row, col, ratio, n_rotation)
        templateBandList += imageList
        
        if (indeks + 1) % 5 == 0:
            print(f"  Processed {indeks + 1}/{len(surveyRowCol)} samples")
    
    print(f"  Total templates created: {len(templateBandList)}")
    
    # ===== Step 7: Template Matching on Tiles =====
    print("\n[8/8] Performing template matching on tiles...")
    matchXYList = process_tile_matching(tiles, tile_metadata, palmRaster, templateBandList)
    print(f"  Total candidate detections: {len(matchXYList)}")
    
    # ===== Step 8: Clustering with BIRCH =====
    print("\n[Final] Performing BIRCH clustering...")
    if len(matchXYList) > 0:
        matchXYArray = np.array(matchXYList)
        brc = Birch(branching_factor=10000, n_clusters=None, threshold=2e-5, compute_labels=True)
        brc.fit(matchXYArray)
        birchPoint = brc.subcluster_centers_
        print(f"  Number of detected trees: {len(birchPoint)}")
    else:
        print("  No detections found!")
        birchPoint = np.array([]).reshape(0, 2)
    
    # ===== Step 9: Visualization and Export =====
    print("\nCreating visualizations...")
    
    if len(matchXYList) > 0:
        # Plot 1: Candidate detections
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(matchXYArray[:, 0], matchXYArray[:, 1], marker='o', c='orangered', s=5, alpha=0.3, label='Candidates')
        show(palmRaster, ax=ax)
        ax.legend()
        ax.set_title('Candidate Tree Detections')
        plt.savefig('Output/candidates.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved: Output/candidates.png")
    
    # Plot 2: Final clustered detections
    if len(birchPoint) > 0:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(birchPoint[:, 0], birchPoint[:, 1], marker='o', color='orangered', s=20, label='Detected Trees')
        show(palmRaster, ax=ax)
        ax.legend()
        ax.set_title(f'Detected Trees (n={len(birchPoint)})')
        plt.savefig('Output/detected_trees.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved: Output/detected_trees.png")
    
    # Export results to CSV
    if len(birchPoint) > 0:
        np.savetxt("Output/birchPoint.csv", birchPoint, delimiter=",", header="X,Y", comments='')
        print("  Saved: Output/birchPoint.csv")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Detected {len(birchPoint)} trees")
    print("=" * 60)
    
    palmRaster.close()

if __name__ == '__main__':
    main()
