import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load raster
with rasterio.open("Steamboat_FirstSnow_DOY.tif") as src:
    first_snow = src.read(1)  # read first band
    profile = src.profile

print("Shape:", first_snow.shape)
print("Min:", np.nanmin(first_snow))
print("Max:", np.nanmax(first_snow))