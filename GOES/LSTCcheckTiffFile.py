from osgeo import osr, gdal
import numpy as np
from netCDF4 import Dataset

# # Open the GeoTIFF file
# dataset = gdal.Open("output_LST.tif")

# if not dataset:
#     print("Failed to open GeoTIFF file")
# else:
#     band = dataset.GetRasterBand(1)
#     array = band.ReadAsArray()

#     print("Min value:", np.min(array))
#     print("Max value:", np.max(array))
#     print("Mean value:", np.mean(array))

#     no_data_value = band.GetNoDataValue()
#     print("NoData value:", no_data_value)

#     unique_values = np.unique(array)
#     print("Unique values in the array:", unique_values)


def check_nc_data_range(path, var_name):
    with Dataset(path, mode="r") as nc:
        data = nc.variables[var_name][:]
        min_value = np.min(data)
        max_value = np.max(data)
        print(f"Min value: {min_value}")
        print(f"Max value: {max_value}")
        print(np.unique(data))


# Update with your NetCDF file path and variable name
check_nc_data_range(
    "/s/lattice-151/a/all/all/all/sustain/data/noaa-goes17/ABI-L2-LSTC/2022/001/01/OR_ABI-L2-LSTC-M6_G17_s20220010101178_e20220010103551_c20220010105239.nc",
    "LST",
)
