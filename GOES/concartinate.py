from osgeo import gdal
import numpy as np
import cv2
import os

"""concartination of all 3 tif bands"""

# def create_multiband_tif(output_path, input_paths):
#     first_ds = gdal.Open(input_paths[0])

#     driver = gdal.GetDriverByName("GTiff")
#     out_ds = driver.Create(
#         output_path,
#         first_ds.RasterXSize,
#         first_ds.RasterYSize,
#         len(input_paths),
#         gdal.GDT_Float32,
#     )

#     if out_ds is None:
#         raise Exception(f"Could not create the output file: {output_path}")

#     for i, input_path in enumerate(input_paths):
#         print(f"Processing {input_path}...")
#         ds = gdal.Open(input_path)
#         band = ds.GetRasterBand(1)
#         data = band.ReadAsArray()
#         out_ds.GetRasterBand(i + 1).WriteArray(data)
#         out_ds.GetRasterBand(i + 1).SetNoDataValue(0)
#         ds = None

#     geotransform = first_ds.GetGeoTransform()
#     out_ds.SetGeoTransform(geotransform)
#     out_ds.SetProjection(first_ds.GetProjection())

#     print("Band values in the combined TIFF:")
#     for i in range(len(input_paths)):
#         band_data = out_ds.GetRasterBand(i + 1).ReadAsArray()
#         print(f"Band {i + 1} values:")
#         print(band_data)

#     out_ds = None
#     print(f"Multi-band TIFF created successfully: {output_path}")


# if __name__ == "__main__":
#     elevation_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030232202/final_elevation.tif"
#     lst_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHash/003/19/030232202//LST.tif"
#     koppen_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppenQuadHash/030232202/koppen.tif"

#     combined_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/combined_elevation_lst_koppen.tif"

#     input_paths = [elevation_tif_path, lst_tif_path, koppen_tif_path]

#     create_multiband_tif(combined_tif_path, input_paths)


"""Concatenation of all 3 TIFF bands after cropping to the bounding box of LST"""


# def get_bounding_box(tif_path):
#     dataset = gdal.Open(tif_path)
#     transform = dataset.GetGeoTransform()

#     # Calculate bounding box
#     min_x = transform[0]  # top-left x
#     max_x = transform[0] + transform[1] * dataset.RasterXSize  # bottom-right x
#     min_y = transform[3] + transform[5] * dataset.RasterYSize  # bottom-right y
#     max_y = transform[3]  # top-left y

#     return (min_x, min_y, max_x, max_y)


# def crop_tif_to_bounding_box(input_tif_path, output_tif_path, bounding_box):
#     min_x, min_y, max_x, max_y = bounding_box

#     gdal.Warp(
#         output_tif_path,
#         input_tif_path,
#         outputBounds=[min_x, min_y, max_x, max_y],
#         format="GTiff",
#     )


# def resize_image(image_path, target_size):
#     """Resize the image to target size using OpenCV."""
#     # Open the image with gdal
#     ds = gdal.Open(image_path)
#     band = ds.GetRasterBand(1)
#     data = band.ReadAsArray()

#     # Resize using OpenCV
#     resized_data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
#     return resized_data


# def create_multiband_tif(output_path, input_paths):
#     first_ds = gdal.Open(input_paths[0])

#     driver = gdal.GetDriverByName("GTiff")
#     out_ds = driver.Create(
#         output_path,
#         first_ds.RasterXSize,
#         first_ds.RasterYSize,
#         len(input_paths),
#         gdal.GDT_Float32,
#     )

#     if out_ds is None:
#         raise Exception(f"Could not create the output file: {output_path}")

#     # Get target dimensions from the first input path (LST)
#     target_width = first_ds.RasterXSize
#     target_height = first_ds.RasterYSize

#     for i, input_path in enumerate(input_paths):
#         print(f"Processing {input_path}...")

#         # Resize image if it does not match the target size
#         resized_data = resize_image(input_path, (target_width, target_height))

#         out_ds.GetRasterBand(i + 1).WriteArray(resized_data)
#         out_ds.GetRasterBand(i + 1).SetNoDataValue(0)

#     geotransform = first_ds.GetGeoTransform()
#     out_ds.SetGeoTransform(geotransform)
#     out_ds.SetProjection(first_ds.GetProjection())

#     print("Band values in the combined TIFF:")
#     for i in range(len(input_paths)):
#         band_data = out_ds.GetRasterBand(i + 1).ReadAsArray()
#         print(f"Band {i + 1} values:")
#         print(band_data)

#     out_ds = None
#     print(f"Multi-band TIFF created successfully: {output_path}")


# if __name__ == "__main__":
#     elevation_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/030232202/final_elevation.tif"
#     lst_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHash/003/19/030232202/LST.tif"
#     koppen_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppenQuadHash/030232202/koppen.tif"

#     bounding_box = get_bounding_box(lst_tif_path)

#     cropped_elevation_path = (
#         "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/cropped_elevation.tif"
#     )
#     cropped_koppen_path = (
#         "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/cropped_koppen.tif"
#     )

#     crop_tif_to_bounding_box(elevation_tif_path, cropped_elevation_path, bounding_box)
#     crop_tif_to_bounding_box(koppen_tif_path, cropped_koppen_path, bounding_box)

#     combined_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/combined_elevation_lst_koppen_2.tif"

#     input_paths = [cropped_elevation_path, lst_tif_path, cropped_koppen_path]

#     create_multiband_tif(combined_tif_path, input_paths)


import os
from osgeo import gdal
import cv2


def resize_image(image_path, target_size):
    """Resize the image to the target size using OpenCV."""
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"Could not open {image_path}")
        return None

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    resized_data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_data


def create_multiband_tif(output_path, input_paths):
    """Create a multi-band TIFF from input images."""
    first_ds = gdal.Open(input_paths[0])
    if first_ds is None:
        print(f"Could not open {input_paths[0]}")
        return

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        first_ds.RasterXSize,
        first_ds.RasterYSize,
        len(input_paths),
        gdal.GDT_Float32,
    )
    if out_ds is None:
        print(f"Could not create the output file: {output_path}")
        return

    target_width, target_height = first_ds.RasterXSize, first_ds.RasterYSize
    for i, input_path in enumerate(input_paths):
        resized_data = resize_image(input_path, (target_width, target_height))
        if resized_data is None:
            print(f"Skipping {input_path} due to resize failure.")
            continue
        out_ds.GetRasterBand(i + 1).WriteArray(resized_data)
        out_ds.GetRasterBand(i + 1).SetNoDataValue(0)

    geotransform = first_ds.GetGeoTransform()
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(first_ds.GetProjection())
    print(f"Multi-band TIFF created: {output_path}")
    out_ds = None


def process_days(
    base_directory,
    elevation_base_path,
    koppen_base_path,
    output_base_path,
):
    """Process LST files and combine with elevation and koppen for a range of days."""
    for day in range(32, 33):
        day_dir = f"{day:03d}"
        day_path = os.path.join(base_directory, day_dir)

        if os.path.isdir(day_path):
            print(f"Processing day directory: {day_dir}")
            for hour_folder in os.listdir(day_path):
                hour_path = os.path.join(day_path, hour_folder)

                if os.path.isdir(hour_path):
                    for quadhash_folder in os.listdir(hour_path):
                        lst_path = os.path.join(hour_path, quadhash_folder, "LST.tif")
                        print(f"Checking for LST file: {lst_path}")

                        if os.path.exists(lst_path):
                            elevation_path = os.path.join(
                                elevation_base_path,
                                quadhash_folder,
                                "final_elevation.tif",
                            )
                            koppen_path = os.path.join(
                                koppen_base_path, quadhash_folder, "koppen.tif"
                            )

                            # Check if elevation and koppen files exist for the same quadhash
                            if os.path.exists(elevation_path) and os.path.exists(
                                koppen_path
                            ):
                                output_path = os.path.join(
                                    output_base_path,
                                    day_dir,
                                    hour_folder,
                                    quadhash_folder,
                                    "3bands_combined.tif",
                                )
                                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                                print(
                                    f"Combining files:\n  LST: {lst_path}\n  Elevation: {elevation_path}\n  Koppen: {koppen_path}"
                                )
                                create_multiband_tif(
                                    output_path, [elevation_path, lst_path, koppen_path]
                                )
                            else:
                                print(
                                    f"Missing elevation or koppen for quadhash {quadhash_folder}"
                                )
                        else:
                            print(f"LST file not found: {lst_path}")


if __name__ == "__main__":
    base_directory = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHash"
    elevation_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash"
    )
    koppen_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppenQuadHash"
    )
    output_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/comdinedQuadHash"
    )

    process_days(
        base_directory,
        elevation_base_path,
        koppen_base_path,
        output_base_path,
    )


print("Process complete")
