import ee

ee.Authenticate(auth_mode="notebook")   # Authenticate Earth Engine, if not already done
ee.Initialize()

# Define the CONUS bounding box as the region of interest
conus_geometry = ee.Geometry.Rectangle(
    [-125, 24, -66.9, 49.4]
)  # Approximate bounds for CONUS

# Load the Landsat 8 Image Collection for 2022, filtering by date and CONUS bounds
landsat2022 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate("2022-01-01", "2022-12-31")
    .filterBounds(conus_geometry)
    .map(lambda image: image.clip(conus_geometry))
)


# Define a function to add NDVI calculation as a band
def add_ndvi(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return image.addBands(ndvi)


# Map the NDVI calculation over the collection
landsat2022 = landsat2022.map(add_ndvi)

# Compute a mean composite for the NDVI over the entire year
landsat_composite = landsat2022.select("NDVI").mean()

# Classify land cover using a threshold (e.g., NDVI > 0.3 for vegetation)
land_cover_type = landsat_composite.gt(0.3).rename("Land_Cover_Type")

# Define an export task for the entire CONUS region
export_task = ee.batch.Export.image.toDrive(
    image=land_cover_type,
    description="Land_Cover_Type_CONUS_2022",
    scale=30,  # 30-meter resolution
    region=conus_geometry.getInfo()["coordinates"],  # CONUS region as bounds
    crs="EPSG:4326",
    maxPixels=1e13,  # To handle large datasets
)

# Start the export task
export_task.start()
