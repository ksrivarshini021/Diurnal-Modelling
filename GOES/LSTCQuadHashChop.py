import socket
import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile, fiona
import geopy.distance
import os
from osgeo import osr, gdal
import json
from pyquadkey2.quadkey import QuadKey
import pyquadkey2
import numpy as np
import rasterio
from shapely.geometry import box


def chop_in_quadhash():
    root_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/states_quads/"
    quadhash_dir = next(
        d
        for d in os.listdir(root_path)
        if os.path.isdir(root_path + d) and d.startswith("quadshape_9_")
    )
    # after runnning create_shp_file funtion, check the created directory and shapefiles. that will be the quadhash dir
    # specific dir within root path that conatains quadhash.shp
    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, "quadhash.shp"))

    in_path = (
        "/s/"
        + socket.gethostname()
        + "/a/all/all/all/sustain/varsh/Python/GOES/TifFolder/001/00.tif"
    )
    out_path = (
        "/s/"
        + socket.gethostname()
        + "/a/all/all/all/sustain/varsh/Python/GOES/"
    )
    count = 0
    total = len(quadhashes)

    for ind, row in quadhashes.iterrows():
        poly, qua = row["geometry"], row["Quadkey"]

        count += 1
        print("Splitting: ", count, "/", total)
        if os.path.exists(out_path + qua + "/nlcd.tif"):
            continue

        os.makedirs(out_path + qua, exist_ok=True)

        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        gdal.Translate(out_path + qua + "/nlcd.tif", in_path, projWin=window)
        if not os.path.exists(out_path + qua + "/nlcd.tif"):
            continue
        x = gdal.Open(out_path + qua + "/nlcd.tif").ReadAsArray()
        if np.min(x) == np.max(x) == 0:
            os.remove(out_path + qua + "/nlcd.tif")
