#!/usr/bin/env python3
"""
Palm Tree Counting Analysis Script
Executes tree detection pipeline using template matching and BIRCH clustering
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from skimage.feature import match_template
import numpy as np
from PIL import Image
from sklearn.cluster import Birch
import os

def main():
    print("=" * 60)
    print("PALM TREE COUNTING ANALYSIS")
    print("=" * 60)
    
    # ===== Step 1: Load Input Data =====
    print("\n[1/7] Loading input shapefile...")
    pointData = gpd.read_file('SHP/sample_point.shp')
    print(f'  CRS of Point Data: {pointData.crs}')
    print(f'  Number of Points: {len(pointData)}')
    print(f'  Geometry Type: {pointData.geom_type.unique()[0]}')
    
    print("\n[2/7] Loading orthophoto raster...")
    palmRaster = rasterio.open('Ortho/example.tif')
    print(f'  CRS of Raster Data: {palmRaster.crs}')
    print(f'  Number of Raster Bands: {palmRaster.count}')
    print(f'  Band Interpretation: {palmRaster.colorinterp}')
    
    # ===== Step 2: Preview Data =====
    print("\n[3/7] Creating preview visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    pointData.plot(ax=ax, color='red', markersize=10, label='Sample Points')
    show(palmRaster, ax=ax)
    ax.legend()
    ax.set_title('Input Data: Points and Orthophoto')
    plt.savefig('Output/preview.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: Output/preview.png")
    
    # ===== Step 3: Extract Point Coordinates to Pixel Positions =====
    print("\n[4/7] Extracting point coordinates to pixel positions...")
    surveyRowCol = []
    
    # Handle MultiPoint geometries
    geom_type = pointData.geom_type.unique()[0]
    if geom_type == "MultiPoint":
        pointData = pointData.explode(index_parts=False)
        print(f"  Expanded MultiPoint to {len(pointData)} individual points")
    
    for index, values in pointData.iterrows():
        geom = values['geometry']
        # Handle both Point and MultiPoint
        if hasattr(geom, 'coords'):
            coords = list(geom.coords)[0]
        else:
            coords = (geom.x, geom.y)
        
        x, y = coords
        row, col = palmRaster.index(x, y)
        if len(surveyRowCol) <= 3:
            print(f"  Point {len(surveyRowCol)+1}: row={row}, col={col}")
        surveyRowCol.append([row, col])
    
    print(f"  Total points processed: {len(surveyRowCol)}")
    
    # ===== Step 4: Extract and Prepare Band Data =====
    print("\n[5/7] Extracting raster bands...")
    redBand = palmRaster.read(1)
    greenBand = palmRaster.read(2)
    blueBand = palmRaster.read(3)
    print(f"  Red band shape: {redBand.shape}")
    print(f"  Green band shape: {greenBand.shape}")
    print(f"  Blue band shape: {blueBand.shape}")
    
    # Create template images with rotation
    print("\n[6/7] Creating template images with rotations...")
    ratio = 25
    n_rotation = 12
    templateBandList = []
    
    for indeks, rowCol in enumerate(surveyRowCol):
        imageList = []
        row = rowCol[0]
        col = rowCol[1]
        
        # Append original band
        imageList.append(greenBand[row-ratio:row+ratio, col-ratio:col+ratio])
        
        # Append rotated images
        templateBandToRotate = greenBand[row-2*ratio:row+2*ratio, col-2*ratio:col+2*ratio]
        rotationList = [i*30 for i in range(1, n_rotation)]
        
        for rotation in rotationList:
            rotatedRaw = Image.fromarray(templateBandToRotate.astype('uint8'))
            rotatedImage = rotatedRaw.rotate(rotation)
            imageList.append(np.asarray(rotatedImage)[ratio:-ratio, ratio:-ratio])
        
        templateBandList += imageList
        if (indeks + 1) % 5 == 0:
            print(f"  Processed {indeks + 1}/{len(surveyRowCol)} samples")
    
    print(f"  Total templates created: {len(templateBandList)}")
    
    # ===== Step 7: Template Matching =====
    print("\n[7/7] Performing template matching across entire image...")
    matchXYList = []
    skipped = 0
    
    for index, templateband in enumerate(templateBandList):
        if index % 50 == 0:
            print(f"  Processing template {index}/{len(templateBandList)}")
        
        # Skip invalid templates (1D or too small)
        if len(templateband.shape) != 2 or templateband.shape[0] < 3 or templateband.shape[1] < 3:
            skipped += 1
            continue
        
        try:
            matchTemplate = match_template(greenBand, templateband, pad_input=True)
            matchTemplateFiltered = np.where(matchTemplate > np.quantile(matchTemplate, 0.9996))
            
            for item in zip(matchTemplateFiltered[0], matchTemplateFiltered[1]):
                x, y = palmRaster.xy(item[0], item[1])
                matchXYList.append([x, y])
        except Exception as e:
            skipped += 1
            continue
    
    print(f"  Total candidate detections: {len(matchXYList)}")
    if skipped > 0:
        print(f"  Skipped invalid templates: {skipped}")
    
    # ===== Step 6: Clustering with BIRCH =====
    print("\n[Final] Performing BIRCH clustering...")
    matchXYArray = np.array(matchXYList)
    brc = Birch(branching_factor=10000, n_clusters=None, threshold=2e-5, compute_labels=True)
    brc.fit(matchXYArray)
    birchPoint = brc.subcluster_centers_
    print(f"  Number of detected trees: {len(birchPoint)}")
    
    # ===== Step 7: Visualization and Export =====
    print("\nCreating visualizations...")
    
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
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(birchPoint[:, 0], birchPoint[:, 1], marker='o', color='orangered', s=20, label='Detected Trees')
    show(palmRaster, ax=ax)
    ax.legend()
    ax.set_title(f'Detected Trees (n={len(birchPoint)})')
    plt.savefig('Output/detected_trees.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: Output/detected_trees.png")
    
    # Export results to CSV
    np.savetxt("Output/birchPoint.csv", birchPoint, delimiter=",", header="X,Y", comments='')
    print("  Saved: Output/birchPoint.csv")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Detected {len(birchPoint)} trees")
    print("=" * 60)
    
    palmRaster.close()

if __name__ == '__main__':
    main()
