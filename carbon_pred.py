import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Paths to NDVI data and output files
ndvi_path = 'test\\ndvi_eastern-WA.tif'
carbon_output_path = 'test\\carbon_potential.tif'

# Biomass estimation model coefficients (example values)
# Replace these with values derived from relevant studies or field data
a = 10000  # Coefficient (e.g., kg/ha)
b = 2      # Exponent

# Carbon fraction in biomass (commonly around 0.47)
carbon_fraction = 0.47

# Open the NDVI raster file
with rasterio.open(ndvi_path) as ndvi_src:
    ndvi = ndvi_src.read(1)
    ndvi_meta = ndvi_src.meta

    # Mask out invalid NDVI values (ensure NDVI is between -1 and 1)
    ndvi = np.where((ndvi >= -1) & (ndvi <= 1), ndvi, np.nan)

    # Apply the biomass estimation model
    # Ensure NDVI values are positive if required by the model
    ndvi_positive = np.maximum(ndvi, 0)
    biomass = a * np.power(ndvi_positive, b)

    # Convert biomass to carbon stock
    carbon_stock = biomass * carbon_fraction

    # Update metadata for the output raster
    carbon_meta = ndvi_meta.copy()
    carbon_meta.update(dtype=rasterio.float32)

    # Write the carbon stock to a new GeoTIFF file
    with rasterio.open(carbon_output_path, 'w', **carbon_meta) as dst:
        dst.write(carbon_stock.astype(rasterio.float32), 1)

# Plot the carbon potential map
plt.figure(figsize=(10, 6))
plt.title('Carbon Potential Map')
carbon_plot = plt.imshow(carbon_stock, cmap='YlGn')
plt.colorbar(carbon_plot, label='Carbon Stock (kg/ha)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()
