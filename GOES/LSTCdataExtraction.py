from goes2go import GOES
from netCDF4 import Dataset
import pandas as pd


# ABI Level 1b Data
G = GOES(satellite=18, product="ABI-L2-LST", domain="C")

startTime = pd.to_datetime("2023-01-01 00:00")
endTime = pd.to_datetime("2023-01-03 00:00")

hourlyRange = pd.date_range(start=startTime, end=endTime, freq="H")

for currentTime in hourlyRange:
    try:
        # Attempt to get files for the current hour
        files = G.timerange(start=currentTime, end=currentTime + pd.Timedelta(hours=1))

        # Check if files are found
        if not files.empty:
            # Download the file
            for file in files["file"]:
                G.download(file)
                print(f"Downloaded: {file}")
        else:
            print(f"No files available for hour {currentTime}")

    except FileNotFoundError as e:
        print(f"File not found for hour {currentTime}: {e}. Skipping...")
        continue

    except Exception as e:
        print(f"Error during download for hour {currentTime}: {e}. Skipping...")

print("Download complete")


# Path to the NetCDF file
# file_path = "/s/lattice-151/a/all/all/all/sustain/data/noaa-goes17/ABI-L2-LSTC/2022/001/01/OR_ABI-L2-LSTC-M6_G17_s20220010101178_e20220010103551_c20220010105239.nc"
# file_path = "/s/parsons/b/others/sustain/varsh/Python/GOES/OR_ABI-L2-LSTC-M6_G16_s20220010001173_e20220010003546_c20220010005176.nc"

# # Open the NetCDF file
# nc_file = Dataset(file_path, mode="r")
# lst_band = nc_file['LST'][:]
# print("bands;", lst_band)
# print('sacalew: ',nc_file['LST'].scale_factor) #0.0025
# print("offser: ", nc_file['LST'].add_offset) # 190.0


# # Print global attributes
# print("Global Attributes:")
# for attr in nc_file.ncattrs():
#     print(f"{attr}: {nc_file.getncattr(attr)}")

# # Print variable names and their attributes
# print("\nVariables and their attributes:")
# for var_name in nc_file.variables:
#     print(f"Variable: {var_name}")
#     var = nc_file.variables[var_name]
#     for attr in var.ncattrs():
#         print(f"  {attr}: {var.getncattr(attr)}")

# nc_file.close()
