import numpy as np
from osgeo import gdal


def open_tif(file_path):
    ds = gdal.Open(file_path)

    print(f"Driver: {ds.GetDriver().ShortName}/{ds.GetDriver().LongName}")
    print(f"Size is {ds.RasterXSize} x {ds.RasterYSize} x {ds.RasterCount}")

    print(f"GeoTransform: {ds.GetGeoTransform()}")
    print(f"Projection: {ds.GetProjection()}")

    return ds


def check_lst_layer(ds):
    if ds is None:
        print("Dataset is not valid.")
        return

    band = ds.GetRasterBand(1)
    if band is None:
        print("LST layer not found.")
        return

    print(f"Band Type: {gdal.GetDataTypeName(band.DataType)}")

    data = band.ReadAsArray()

    min_value = np.min(data)
    max_value = np.max(data)
    print(f"Min/Max LST values: {min_value}, {max_value}")

    print("Sample LST values:")
    sample_values = data[:5, :5]
    print(sample_values)


file_path = "lst_output.tif"
ds = open_tif(file_path)

check_lst_layer(ds)

ds = None
