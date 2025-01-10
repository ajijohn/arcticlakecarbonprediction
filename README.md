# Arctic Lakes Carbon Prediction Project

Welcome to the Arctic Lakes Carbon Prediction Project! This project aims to estimate the carbon potential of Arctic lakes by analyzing satellite imagery and calculating the Normalized Difference Vegetation Index (NDVI). By correlating NDVI values with biomass and carbon stock estimations, we can better understand carbon dynamics in these critical ecosystems.

## Table of Contents

- Introduction
- Project Structure
- Features
- Requirements
- Installation
- Usage
- Data Acquisition
- Methodology
- Results
- Contributing
- License
- Contact

## Introduction

Arctic lakes play a significant role in the global carbon cycle, especially in the context of climate change and permafrost thaw. This project provides tools and scripts to:

- Calculate NDVI from satellite imagery.
- Estimate biomass from NDVI values.
- Convert biomass estimates to carbon stock.
- Visualize and analyze the spatial distribution of carbon potential.

## Project Structure

```
arcticlakecarbonprediction/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── calculate_ndvi.py
│   ├── estimate_carbon.py
│   └── utils.py
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── ndvi_maps/
│   └── carbon_maps/
├── README.md
├── requirements.txt
└── LICENSE
```

## Features

- Predict lake boundaries
- NDVI Calculation: Compute NDVI from red and near-infrared (NIR) satellite imagery bands.
- Biomass Estimation: Apply empirical models to estimate vegetation biomass from NDVI.
- Carbon Stock Calculation: Convert biomass estimates to carbon stock using standard conversion factors.
- Data Visualization: Generate maps and plots to visualize NDVI and carbon distribution.
- Scalability: Process large datasets and handle multiple satellite images.

## Requirements

- Python 3.7 or higher
- Libraries:
- rasterio
- numpy
- matplotlib
- pandas
- geopandas (optional for vector data handling)
- Satellite Imagery:
- Access to red and NIR bands from satellite data (e.g., Landsat 8, Sentinel-2)

## Installation

 1. Clone the Repository

```
git clone https://github.com/ajijohn/arcticlakecarbonprediction.git
cd arcticlakecarbonprediction
```

## Create a Virtual Environment (Optional)

```
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

 3. Install Dependencies

```
pip install -r requirements.txt
```

## Usage

1. Prepare the Data

- Download Satellite Imagery:
- Obtain red and NIR bands for your area of interest.
- Place the files in the data/raw/ directory.

2. Predict lakes
3. Calculate NDVI

Run the NDVI calculation script:

```
python scripts/calculate_ndvi.py --red data/raw/red_band.tif --nir data/raw/nir_band.tif --output data/processed/ndvi.tif
```

- Arguments:
- --red: Path to the red band image.
- --nir: Path to the NIR band image.
- --output: Path to save the NDVI output.

3. Estimate Carbon Stock

Run the carbon estimation script:

```
python scripts/estimate_carbon.py --ndvi data/processed/ndvi.tif --output data/processed/carbon_stock.tif
```

- Arguments:
- --ndvi: Path to the NDVI image.
- --output: Path to save the carbon stock output.

4. Visualize Results

- Use the provided Jupyter Notebook notebooks/analysis.ipynb to visualize and analyze the results.
- Launch Jupyter Notebook:

```
jupyter notebook notebooks/analysis.ipynb
```

## Data Acquisition

Satellite Data Sources

- Landsat 8:
  - USGS EarthExplorer
  - Red Band: Band 4
  - NIR Band: Band 5
- Sentinel-2:
  - Copernicus Open Access Hub
  - Red Band: Band 4
  - NIR Band: Band 8

## Preprocessing Steps

- Ensure Spatial Alignment:
  - Bands must be co-registered and have the same spatial resolution.
- Cloud Masking:
  - Apply cloud masks if necessary to remove cloud-covered pixels.
- Data Formats:
  - Supported formats include GeoTIFF and other raster data formats compatible with rasterio.

## Methodology

1. Predict lakes
2. NDVI Calculation

NDVI is calculated using the formula:

- Purpose: NDVI indicates the presence and condition of vegetation.
- Implementation: The calculate_ndvi.py script reads the red and NIR bands and computes NDVI for each pixel.

3. Biomass Estimation

- Model Used:
- Parameters:
- Coefficients a and b are derived from relevant studies or field data specific to Arctic vegetation.

4. Carbon Stock Calculation

- With NDVI - Use Conversion: Carbon Stock = Biomass × Carbon Fraction 
- Carbon Fraction: Typically around 0.47 (i.e., 47% of biomass is carbon).
- Basic carbon map from satellite-derived lake predictions
	•	Calculate the area of each detected lake (in square meters or kilometers).
	•	Using global estimates of Methane emissions: ~0.1–1.0 g CH₄/m²/day (varies by lake size, temperature, trophic status).
	•	Sediment carbon sequestration: ~5–30 g C/m²/year (varies by lake type).
	•	Assign average carbon flux or stock values to each lake area based on size, location, or type (see Raymond et al., 2013).
	•	Carbon Flux Estimates: Derived from literature on lake emissions (e.g., Raymond et al., 2013; Bastviken et al., 2011)

5. Data Visualization

- Maps and Plots:
- Generate spatial maps of NDVI and carbon stock.
- Analyze the spatial distribution and identify areas with high carbon potential.

## Results

- NDVI Maps:
- Located in results/ndvi_maps/
- Carbon Stock Maps:
- Located in results/carbon_maps/
- Analysis Reports:
- Findings and insights documented in notebooks/analysis.ipynb

## Contributing

We welcome contributions from the community! If you’d like to contribute:

 1. Fork the Repository
 2. Create a Feature Branch

```
git checkout -b feature/your-feature-name
```

 3. Commit Your Changes

```
git commit -m "Add your message here"
```

 4. Push to Your Fork

```
git push origin feature/your-feature-name
```

 5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Project Maintainer:  Aji John
- GitHub: ajijohn

## Acknowledgments

- Data Providers:
- NASA
- ESA
- USGS
- References:
- Raynolds, M. K., et al. (2012). “A new estimate of tundra-biome phytomass from trans-Arctic field data and AVHRR NDVI.” Remote Sensing Letters.
- Mishra, U., & Riley, W. J. (2012). “Alaska carbon stocks: spatial variability and dependence on environmental factors.” Biogeosciences.

## Troubleshooting

- Common Issues:
- Module Not Found: Ensure all dependencies are installed via pip install -r requirements.txt.
- Data Alignment Errors: Check that input raster files have the same dimensions and coordinate reference system (CRS).
- Empty Outputs: Verify that the input data paths are correct and that the images are not corrupted.
- Support:
- Open an issue on GitHub for assistance.
- Consult the documentation of the libraries used.

## Future Work

- Integration of Machine Learning Models:
- Implement advanced models for biomass estimation.
- Time-Series Analysis:
- Analyze changes over time using historical satellite imagery.
- Incorporate Belowground Biomass:
- Extend the methodology to estimate total ecosystem carbon.

Thank you for your interest in the Arctic Lakes Carbon Prediction Project.
