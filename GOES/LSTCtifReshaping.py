from osgeo import gdal
import os


def resample_image(input_tif_path, output_tif_path, target_pixel_size_km):
    """
    Resamples the TIFF image to the specified resolution in kilometers per pixel.
    """
    dataset = gdal.Open(input_tif_path)

    km_to_deg = 1 / 111.32
    target_pixel_size_deg = target_pixel_size_km * km_to_deg

    # Resample using gdal.Warp
    gdal.Warp(
        output_tif_path,
        dataset,
        xRes=target_pixel_size_deg,
        yRes=target_pixel_size_deg,
        resampleAlg="bilinear",
        dstSRS=dataset.GetProjection(),
    )
    print(f"Resampled TIFF saved at {output_tif_path}")


def resize_to_target_dimensions(
    input_tif_path, output_tif_path, target_width, target_height
):
    """
    Resizes the TIFF image to the specified width and height.
    """
    dataset = gdal.Open(input_tif_path)

    # Resize using gdal.Warp
    gdal.Warp(
        output_tif_path,
        dataset,
        width=target_width,
        height=target_height,
        resampleAlg="bilinear",
        dstSRS=dataset.GetProjection(),
    )
    print(f"Resized TIFF saved at {output_tif_path}")


# Input and output base directories
input_base_directory = (
    "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHashTest/"
)
output_base_directory = (
    "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHashTest32x32/"
)

# Iterate through the 
# directory structure
for day in range(31, 32):
    day_dir = f"{day:03d}"
    day_path = os.path.join(input_base_directory, day_dir)
    output_day_path = os.path.join(output_base_directory, day_dir)

    if os.path.isdir(day_path):
        for hour_folder in os.listdir(day_path):
            hour_path = os.path.join(day_path, hour_folder)
            output_hour_path = os.path.join(output_day_path, hour_folder)

            if os.path.isdir(hour_path):
                for quadhash_folder in os.listdir(hour_path):
                    quadhash_path = os.path.join(hour_path, quadhash_folder)
                    output_quadhash_path = os.path.join(
                        output_hour_path, quadhash_folder
                    )

                    if os.path.isdir(quadhash_path):
                        # Ensure the output directory exists
                        os.makedirs(output_quadhash_path, exist_ok=True)

                        for file_name in os.listdir(quadhash_path):
                            if file_name.endswith(".tif"):
                                input_tif_path = os.path.join(quadhash_path, file_name)
                                resampled_tif_path = os.path.join(
                                    output_quadhash_path, file_name
                                )

                                # Step 1: Resample the TIFF file to 2 km resolution
                                resample_image(
                                    input_tif_path,
                                    resampled_tif_path,
                                    target_pixel_size_km=2.0,
                                )

                                # Step 2: Resize the resampled TIFF file to 32x32 pixels
                                resized_tif_path = os.path.join(
                                    output_quadhash_path, file_name
                                )
                                resize_to_target_dimensions(
                                    resampled_tif_path,
                                    resized_tif_path,
                                    target_width=32,
                                    target_height=32,
                                )

print("Processing complete.")


""" FOR ELEVATION RESHAPING"""


# def resample_elevation(input_tif_path, output_tif_path, target_resolution_km):
#     dataset = gdal.Open(input_tif_path)

#     km_to_deg = 1 / 111.32
#     target_pixel_size = target_resolution_km * km_to_deg

#     gdal.Warp(
#         output_tif_path,
#         dataset,
#         xRes=target_pixel_size,
#         yRes=target_pixel_size,
#         resampleAlg="bilinear",
#     )

#     print(f"Resampled elevation TIFF saved at {output_tif_path}")


# def resize_to_target_dimensions(
#     input_tif_path, output_tif_path, target_width, target_height
# ):
#     dataset = gdal.Open(input_tif_path)

#     crs = dataset.GetProjection()

#     gdal.Warp(
#         output_tif_path,
#         dataset,
#         width=target_width,
#         height=target_height,
#         resampleAlg="bilinear",
#         dstSRS=crs,
#     )

#     print(f"Resized elevation TIFF saved at {output_tif_path}")


# base_directory = (
#     "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/"
# )

# for folder_name in os.listdir(base_directory):
#     folder_path = os.path.join(base_directory, folder_name)

#     if os.path.isdir(folder_path) and len(folder_name) == 9:
#         input_elevation_path = os.path.join(folder_path, "final_elevation_30m.tif")

#         if os.path.exists(input_elevation_path):
#             # Step 1: Resample the original elevation TIFF to 2 km resolution
#             output_elevation_path = os.path.join(folder_path, "final_elevation_2km.tif")
#             resample_elevation(
#                 input_elevation_path, output_elevation_path, target_resolution_km=2.0
#             )

#             # Step 2: Resize the 2 km TIFF to 32x32 pixels
#             output_resized_elevation_path = os.path.join(
#                 folder_path, "final_elevation.tif"
#             )
#             resize_to_target_dimensions(
#                 output_elevation_path,
#                 output_resized_elevation_path,
#                 target_width=32,
#                 target_height=32,
#             )
#         else:
#             print(f"Input file not found: {input_elevation_path}")

# print("Processing complete.")


"""for koppen reshaping"""


# def resample_image(input_tif_path, output_tif_path, target_pixel_size_km):
#     dataset = gdal.Open(input_tif_path)

#     transform = dataset.GetGeoTransform()
#     crs = dataset.GetProjection()

#     km_to_deg = 1 / 111.32
#     target_pixel_size_deg = target_pixel_size_km * km_to_deg

#     # gdal.Warp to resample image
#     gdal.Warp(
#         output_tif_path,
#         dataset,
#         xRes=target_pixel_size_deg,
#         yRes=target_pixel_size_deg,
#         resampleAlg="bilinear",
#         dstSRS=crs,
#     )

#     print(f"Resampled TIFF saved at {output_tif_path}")


# def resize_to_target_dimensions(
#     input_tif_path, output_tif_path, target_width, target_height
# ):
#     dataset = gdal.Open(input_tif_path)

#     crs = dataset.GetProjection()

#     gdal.Warp(
#         output_tif_path,
#         dataset,
#         width=target_width,
#         height=target_height,
#         resampleAlg="bilinear",
#         dstSRS=crs,
#     )

#     print(f"Resized TIFF saved at {output_tif_path}")


# base_directory_koppen = (
#     "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppenQuadHash"
# )

# for folder_name in os.listdir(base_directory_koppen):
#     folder_path = os.path.join(base_directory_koppen, folder_name)

#     if os.path.isdir(folder_path) and len(folder_name) == 9:
#         koppen_tif_path = os.path.join(folder_path, "koppen.tif")

#         if os.path.isfile(koppen_tif_path):
#             # Step 1: Resample the original Koppen TIFF to 2 km resolution
#             output_2km_tif_path = os.path.join(folder_path, "koppen_2km.tif")
#             resample_image(
#                 koppen_tif_path,
#                 output_2km_tif_path,
#                 target_pixel_size_km=2.0,
#             )

#             # Step 2: Resize the 2 km Koppen TIFF to 32x32 pixels
#             output_32x32_tif_path = os.path.join(folder_path, "koppen.tif")
#             resize_to_target_dimensions(
#                 output_2km_tif_path,
#                 output_32x32_tif_path,
#                 target_width=32,
#                 target_height=32,
#             )
#         else:
#             print(f"koppen.tif not found in {folder_path}")

# print("Processing complete.")
