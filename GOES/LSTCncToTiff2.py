from netCDF4 import Dataset
from osgeo import osr, gdal
import os

def get_goes_projection(dataset):
    """Extract GOES-18 projection and return OSR SpatialReference."""
    projection_var_name = "goes_imager_projection"
    if projection_var_name in dataset.variables:
        proj_var = dataset.variables[projection_var_name]
        proj_params = proj_var.__dict__
    else:
        proj_params = dataset.__dict__

    proj4_string = "+proj=geos"
    if "semi_major_axis" in proj_params:
        proj4_string += f" +a={proj_params['semi_major_axis']}"
    if "inverse_flattening" in proj_params:
        f = 1 / proj_params["inverse_flattening"]
        proj4_string += f" +f={f}"
    if "longitude_of_projection_origin" in proj_params:
        proj4_string += f" +lon_0={proj_params['longitude_of_projection_origin']}"
    if "latitude_of_projection_origin" in proj_params:
        proj4_string += f" +lat_0={proj_params['latitude_of_projection_origin']}"
    if "perspective_point_height" in proj_params:
        proj4_string += f" +h={proj_params['perspective_point_height']}"
    if "sweep_angle_axis" in proj_params:
        proj4_string += f" +sweep={proj_params['sweep_angle_axis']}"

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromProj4(proj4_string)
    return spatial_ref

def convert_nc_to_tif(nc_file, output_path, dstSRS, pixel_size_km=2.0):
    """Convert GOES LST NetCDF to GeoTIFF in Kelvin, keeping fill values intact."""
    var_name = "LST"
    dataset = Dataset(nc_file, mode="r")
    goes_proj = get_goes_projection(dataset)

    lst_var = dataset.variables[var_name]

    # Read scale factor and fill value from metadata
    scale_factor = getattr(lst_var, "scale_factor", 1.0)
    fill_value = getattr(lst_var, "_FillValue", 65535)

    # Open the NetCDF variable via GDAL
    band = gdal.Open(f'NETCDF:"{nc_file}":{var_name}')

    tmp_tif = "./tmp_goes.tif"

    # Scale valid values to Kelvin, leave fill_value unchanged
    gdal.Translate(
        tmp_tif,
        band,
        scaleParams=[[0, fill_value - 1, 0, (fill_value - 1) * scale_factor]],
        outputType=gdal.GDT_Float32
    )

    # Reproject to target SRS
    deg_per_pixel = pixel_size_km / 111.0
    gdal.Warp(
        output_path,
        tmp_tif,
        dstSRS=dstSRS,
        xRes=deg_per_pixel,
        yRes=deg_per_pixel,
        resampleAlg=gdal.GRA_NearestNeighbour
    )

    dataset.close()
    os.remove(tmp_tif)

def process_day(day_input_path, output_base_path, targetPrj, resolution_km=2.0):
    """Process one day's directory containing hourly NC files."""
    if not os.path.isdir(day_input_path):
        print(f"Day directory not found: {day_input_path}")
        return

    for hour in sorted(os.listdir(day_input_path)):
        hour_input_path = os.path.join(day_input_path, hour)
        if not os.path.isdir(hour_input_path):
            continue

        nc_files = [f for f in os.listdir(hour_input_path) if f.endswith(".nc")]
        if not nc_files:
            print(f"No .nc files found in {hour_input_path}. Skipping...")
            continue

        output_hour_path = os.path.join(output_base_path, "001")
        os.makedirs(output_hour_path, exist_ok=True)

        output_path = os.path.join(output_hour_path, f"{hour}.tif")
        nc_file = os.path.join(hour_input_path, nc_files[0])
        print(f"Processing {nc_file} => {output_path}")
        convert_nc_to_tif(nc_file, output_path, targetPrj, resolution_km)

if __name__ == "__main__":
    input_day_path = "/s/parsons/b/others/sustain/data/noaa-goes18/ABI-L2-LSTC/2023/001/"
    output_base_path = "/s/parsons/b/others/sustain/varsh/Python/GOES/test2023"

    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    print(f"Target Projection: {target_srs.ExportToProj4()}")

    process_day(input_day_path, output_base_path, target_srs)
