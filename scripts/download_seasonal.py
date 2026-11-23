
import cdsapi
from pathlib import Path

# --------------------------------------------
# Project directories
# --------------------------------------------

root = Path(__file__).resolve().parents[1]

output_dir = root / "inputs" / "seasonal_forecasts"
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# ECMWF SEAS5 forecast settings
# --------------------------------------------

dataset = "seasonal-original-single-levels"

# October 1, 2022 initialization
year = "2022"
month = "10"
day = "01"

# Daily forecast lead times, 1 to 180 days
# 24, 48, 72, ..., 4320 hours
lead_times = [str(hour) for hour in range(24, 4321, 24)]

request = {
    "originating_centre": "ecmwf",
    "system": "51",

    "variable": [
        "maximum_2m_temperature_in_the_last_24_hours",
        "minimum_2m_temperature_in_the_last_24_hours",
        "total_precipitation",
    ],

    "year": [year],
    "month": [month],
    "day": [day],

    "leadtime_hour": lead_times,

    # North, West, South, East
    # Covers the five study states with a buffer
    "area": [50, -118, 36, -101],

    "data_format": "grib",
}

# --------------------------------------------
# Output file
# --------------------------------------------

target = output_dir / "ecmwf_202210_full_winter.grib"

# Avoid accidentally downloading twice
if target.exists():
    print(f"File already exists: {target}")
    print(f"Size: {target.stat().st_size / 1e6:.2f} MB")
    raise SystemExit(0)

# --------------------------------------------
# Download
# --------------------------------------------

print("=" * 55)
print("ECMWF SEAS5 FULL WINTER DOWNLOAD")
print("=" * 55)

print(f"Initialization: {year}-{month}-{day}")
print(f"Forecast days: {len(lead_times)}")
print("Ensemble members: 51 expected")
print("Variables: minimum temperature, maximum temperature, precipitation")
print("Forecast period: October 2, 2022 - March 30, 2023")
print(f"Output: {target}")

print("\nSubmitting request to Copernicus...")

client = cdsapi.Client()
client.retrieve(dataset, request, str(target))

print("\nDownload complete!")
print(f"Saved to: {target}")
print(f"File size: {target.stat().st_size / 1e6:.2f} MB")

