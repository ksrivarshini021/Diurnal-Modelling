import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import numpy as np


# def resample_tif_to_64x64(input_path, output_path):
#     with rasterio.open(input_path) as src:
#         target_width, target_height = 64, 64

#         # Calculate scaling factors
#         scale_x = src.width / target_width
#         scale_y = src.height / target_height

#         # Modify the transform with scaling for width and height
#         transform = src.transform * Affine.scale(scale_x, scale_y)

#         # Update profile with new width, height, and transform
#         profile = src.profile
#         profile.update(width=target_width, height=target_height, transform=transform)

#         # Resample the data to the new shape
#         data = src.read(
#             out_shape=(src.count, target_height, target_width),
#             resampling=Resampling.bilinear,  # Try other methods if needed
#         )

#         # Write the resampled data to a new .tif file
#         with rasterio.open(output_path, "w", **profile) as dst:
#             dst.write(data)
#         print(f"Resampled image saved at: {output_path}")


# # Example usage
# input_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030223330/final_elevation_30m.tif"
# output_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030223330/final_elevation_30m_resampled_64x64.tif"

# resample_tif_to_64x64(input_path, output_path)


def print_tif_dimensions(tif_path):
    with rasterio.open(tif_path) as tif:
        width, height = tif.width, tif.height
        resolution = 30

        # real_world_width = width * resolution
        # real_world_height = height * resolution

        real_world_width = width * resolution / 1000
        real_world_height = height * resolution / 1000

        print(f"Dimensions of {tif_path}:")
        print(f"Width: {width} pixels, Height: {height} pixels")
        print(f"Real-World Width: {real_world_width:.2f} km")
        print(f"Real-World Height: {real_world_height:.2f} km")
        print(f"Number of Bands: {tif.count}")


tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030223330/final_elevation_30m.tif"
# tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHash/120/00/030232110/nlcd.tif"
print_tif_dimensions(tif_path)


# import rasterio


# def get_tif_details(tif_path):
#     with rasterio.open(tif_path) as tif:
#         width, height = tif.width, tif.height
#         transform = tif.transform

#         pixel_width = transform[0]
#         pixel_height = -transform[4]
#         real_world_width = width * pixel_width / 1000
#         real_world_height = height * pixel_height / 1000

#         num_bands = tif.count

#         # Print the results
#         print(f"Dimensions of {tif_path}:")
#         print(f"Width: {width} pixels, Height: {height} pixels")
#         print(f"Real-World Width: {real_world_width:.2f} km")
#         print(f"Real-World Height: {real_world_height:.2f} km")
#         print(f"Number of Bands: {num_bands}")


# # tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030223330/final_elevation_30m_resampled_64x64.tif"
# tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030223330/final_elevation_30m.tif"
# # get_tif_resolution(tif_path)
