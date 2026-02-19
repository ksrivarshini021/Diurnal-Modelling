import rasterio
import numpy as np
import csv
from math import cos, radians

def calculate_resolution_in_km(pixel_size, latitude):
    """Calculate the resolution in kilometers per pixel."""
    km_per_degree_latitude = 111  
    km_per_degree_longitude = 111 * abs(cos(radians(latitude)))  

    x_km = pixel_size[0] * km_per_degree_longitude
    y_km = pixel_size[1] * km_per_degree_latitude

    return x_km, y_km


def compute_longitudes_from_transform(transform, width, height):
    """Compute the longitude for each pixel in the raster."""
    lon_values = np.array([transform.c + transform.a * x for x in range(width)])
    longitudes = np.tile(lon_values, (height, 1))
    return longitudes


def compute_latitudes_from_transform(transform, width, height):
    """Compute the latitude for each pixel in the raster."""
    lat_values = np.array([transform.f + transform.e * y for y in range(height)])
    latitudes = np.tile(lat_values, (width, 1)).T  
    return latitudes


def convert_solar_to_local_time(solar_time, longitudes):
    """Convert solar time to local time using longitude adjustment."""
    return solar_time + (longitudes / 15) 


def get_lat_lon_from_pixel(transform, row, col):
    """Get the latitude and longitude of a pixel from its row and column."""
    longitude = transform.c + transform.a * col
    latitude = transform.f + transform.e * row
    return latitude, longitude


def print_tif_metadata(tif_path):
    with rasterio.open(tif_path) as src:
        print(f"Metadata for {tif_path}:")
        print(f"  Width: {src.width}")
        print(f"  Height: {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Pixel Size: {src.res}")
        print(f"  Number of Bands: {src.count}")
        print(f"  Data Type: {src.dtypes}")
        print(f"  Bounds: {src.bounds}")
        print("-" * 40)


def generate_local_time_table(image_path, output_csv):
    with rasterio.open(image_path) as src:
        solar_time = src.read(1)  
        transform = src.transform
        crs = src.crs

    height, width = solar_time.shape

    longitudes = compute_longitudes_from_transform(transform, width, height)
    latitudes = compute_latitudes_from_transform(transform, width, height)

    local_time = convert_solar_to_local_time(solar_time, longitudes)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Latitude', 'Longitude', 'Local Time (Hours)']) 
        
        for row in range(height):
            for col in range(width):
                latitude, longitude = get_lat_lon_from_pixel(transform, row, col)
                time = local_time[row, col]
                writer.writerow([latitude, longitude, time])


if __name__ == "__main__":
    image_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHashTest32x32/001/19/030232110/001_19_030232110.tif"  
    output_csv = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHashTest32x32/001/19/030232110/local_time_table.csv"  
    print_tif_metadata(image_path)
    generate_local_time_table(image_path, output_csv)

    print(f"Local time table has been saved to {output_csv}")
