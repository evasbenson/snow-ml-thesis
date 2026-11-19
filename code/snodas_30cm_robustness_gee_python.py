"""Python translation of the 30 cm SNODAS GEE extraction.
Install: pip install earthengine-api
Authenticate once: earthengine authenticate
Run: python snodas_30cm_robustness_gee_python.py
IMPORTANT: verify source band and depth units before launching export.
"""
import ee

# If required: ee.Initialize(project='YOUR_EARTH_ENGINE_CLOUD_PROJECT')
ee.Initialize()

FIPS = [
    '08037','08117','08097','08107','08113','08049','08019','08051','08067',
    '49035','49043','49051','49003','49057','56039','30031','30029','30067',
    '16013','16055','16017','16081',
]
COUNTIES = ee.FeatureCollection('TIGER/2018/Counties').filter(
    ee.Filter.inList('GEOID', FIPS)
)
SNODAS = ee.ImageCollection('projects/climate-engine/snodas/daily')
ELEV_MASK = ee.Image('USGS/SRTMGL1_003').select('elevation').gte(2000)

# MUST verify these against image metadata and the original 10 cm extraction.
DEPTH_BAND = 'snow_depth'  # Placeholder: inspect printed bands below.
DEPTH_TO_METERS = 1.0     # Placeholder: 0.001 for mm, 1 for meters, etc.

print('Available bands:', SNODAS.first().bandNames().getInfo())
print('First image properties:', SNODAS.first().toDictionary().getInfo())
answer = input('Verified DEPTH_BAND and DEPTH_TO_METERS against source and original extraction? Type YES: ')
if answer != 'YES':
    raise SystemExit('Export canceled; verify band name and units first.')


def winter_features(year):
    year = ee.Number(year)
    start = ee.Date.fromYMD(year.subtract(1), 11, 1)
    end = ee.Date.fromYMD(year, 5, 1)

    def snow_indicator(image):
        depth_m = image.select(DEPTH_BAND).multiply(DEPTH_TO_METERS)
        return depth_m.gt(0.30).unmask(0).rename('snow30').toFloat()

    count_image = SNODAS.filterDate(start, end).map(snow_indicator).sum().updateMask(ELEV_MASK)

    def county_feature(county):
        mean = count_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=county.geometry(),
            scale=1000,
            maxPixels=1e9,
            tileScale=4,
        ).get('snow30')
        return ee.Feature(None, {
            'fips': county.get('GEOID'),
            'county': county.get('NAME'),
            'year': year,
            'snow_days_30cm_2000m': mean,
        })

    return COUNTIES.map(county_feature)


result = ee.FeatureCollection(ee.List.sequence(2004, 2023).map(winter_features)).flatten()
print('Expected exported county-years:', result.size().getInfo())
task = ee.batch.Export.table.toDrive(
    collection=result,
    description='county_snodas_30cm_2000m',
    folder='GEE_exports',
    fileNamePrefix='county_snodas_30cm_2000m',
    fileFormat='CSV',
)
task.start()
print('Export started. Task ID:', task.id)
print('Monitor at https://code.earthengine.google.com/tasks')
