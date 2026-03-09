# Tree Counting Using Template Matching and Clustering

This project implements a semi-automatic method for detecting and counting trees from aerial or satellite imagery using image processing and clustering techniques. The workflow combines **template matching**, **statistical filtering**, and **clustering (BIRCH)** to identify individual tree locations from raster imagery.

The main objective of this project is to provide a lightweight alternative to deep learning approaches for tree detection using classical computer vision and machine learning methods.

---

# Overview

The detection pipeline consists of the following steps:

1. Load raster imagery containing tree canopy data.
2. Extract RGB bands from the raster dataset.
3. Generate candidate tree locations using **template matching**.
4. Apply statistical filtering using **quantile thresholding** to keep only the strongest matches.
5. Cluster nearby detections using **BIRCH clustering** to represent individual trees.
6. Visualize detected trees and export the results as spatial data.

This method is particularly useful for:

* Plantation monitoring
* Tree inventory estimation
* Agricultural mapping
* Forestry analysis

---

# Project Structure

```
tree-counting/
│
├── tree counting.ipynb      # Main notebook for detection workflow
├── SHP/                     # Input shapefiles (sample points or AOI)
│   └── sample.shp
│
├── data/                    # Raster imagery
│   └── image.tif
│
├── output/                  # Generated outputs
│   ├── detected_points.shp
│   └── visualization.png
│
└── README.md
```

---

# Methodology

## 1. Raster Processing

The raster image is loaded and its bands are extracted.

```python
redBand = palmRaster.read(1)
greenBand = palmRaster.read(2)
blueBand = palmRaster.read(3)
```

These bands are used for visualization and feature extraction.

---

## 2. Template Matching

Template matching is used to locate potential tree crowns by comparing a predefined template with the raster image.

The output is a similarity matrix where higher values indicate stronger matches.

---

## 3. Statistical Filtering

To reduce false detections, only the highest similarity scores are kept using quantile filtering.

```python
matchTemplateFiltered = np.where(
    matchTemplate > np.quantile(matchTemplate, 0.9996)
)
```

This keeps only the top **0.04%** of candidate detections.

---

## 4. Clustering

Since template matching often detects multiple points per tree, clustering is applied to merge nearby detections.

This project uses **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**.

```python
from sklearn.cluster import Birch
```

Each resulting cluster represents a single tree location.

---

## 5. Visualization

Detected points are visualized on top of the raster image to evaluate detection quality.

Sample inspection is performed by zooming into multiple candidate points.

---

# Requirements

The project requires the following Python libraries:

```
numpy
opencv-python
matplotlib
rasterio
geopandas
scikit-learn
scipy
```

Install them using:

```
pip install numpy opencv-python matplotlib rasterio geopandas scikit-learn scipy
```

---

# Usage

1. Clone the repository

```
git clone https://github.com/yourusername/tree-counting.git
cd tree-counting
```

2. Prepare your data

Place your raster imagery in the `data` folder and shapefiles in the `SHP` directory.

3. Run the notebook

Open and execute:

```
tree counting.ipynb
```

4. Review the outputs

The detected tree points and visualizations will be generated in the output directory.

---

# Limitations

This approach may produce false detections in areas with similar patterns such as:

* roads
* shadows
* field boundaries

Additional filtering techniques such as vegetation indices (NDVI) or spatial masks can improve results.

---

# Future Improvements

Possible enhancements include:

* NDVI based vegetation masking
* Road masking using GIS layers
* Non-Maximum Suppression for detection refinement
* Deep learning integration for improved detection accuracy
* Automated parameter tuning

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This project uses open-source geospatial and machine learning libraries including:

* Rasterio
* GeoPandas
* OpenCV
* Scikit-learn
