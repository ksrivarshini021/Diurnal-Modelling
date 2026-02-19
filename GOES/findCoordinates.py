from netCDF4 import Dataset
import numpy as np


def calculate_degrees(file_id):
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables["y"][:]  # N/S elevation angle in radians
    projection_info = file_id.variables["goes_imager_projection"]
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)
    return lon, lat


# Path to the NetCDF file
file_path = "/s/lattice-151/a/all/all/all/sustain/data/noaa-goes16/ABI-L2-MCMIPC/2022/001/00/OR_ABI-L2-MCMIPC-M6_G16_s20220010001173_e20220010003557_c20220010004046.nc"

# Open the file using the netCDF4 library
file_id = Dataset(file_path)

# Call function to calculate latitude and longitude from GOES ABI fixed grid projection data
lon, lat = calculate_degrees(file_id)

# Print the max and min latitude and longitude values
print("The maximum latitude value is", np.max(lat), "degrees")
print("The minimum latitude value is", np.min(lat), "degrees")
print("The maximum longitude value is", np.max(lon), "degrees")
print("The minimum longitude value is", np.min(lon), "degrees")
