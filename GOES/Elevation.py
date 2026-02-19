import rasterio
from dem_stitcher.stitcher import stitch_dem
import numpy as np
import socket
import os
import geopandas as gpd
import pyproj
mins = 100000
maxs = -9999


def download_dem(out_path, poly):
    global mins
    global maxs
    try:
        X, p = stitch_dem(
            list(poly.bounds),
            dem_name="glo_30",
            dst_ellipsoidal_height=False,
            dst_area_or_point="Area",
        )
    except Exception as e:
        print("ERROR Downloading: ", out_path.split("split_9")[1], e)
        return

    if np.min(X) < mins:
        mins = np.min(X)
    if np.max(X) > maxs:
        maxs = np.max(X)

    save_as_tif(X, p, out_path)


def save_as_tif(X, p, out_path):
    with rasterio.open(out_path, "w", **p) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT="Area")


def chop_in_quadhash():
    out_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash2/"
    )

    root_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/states_quads(09)/"
    quadhash_dir = next(
        d
        for d in os.listdir(root_path)
        if os.path.isdir(root_path + d) and d.startswith("quadshape_9_")
    )

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, "quadhash.shp"))

    total = len(quadhashes)
    count = 0

    for ind, row in quadhashes.iterrows():
        poly, qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        count += 1
        print("Processing: ", count, "/", total, qua, flush=True)
        if os.path.exists(out_path + qua + "/final_elevation_30m.tif"):
            continue
        download_dem(out_path + qua + "/final_elevation_30m.tif", poly)

    print("-------  Min dem value in state: ", mins, "------ max: ", maxs)


def remove_empty_folders():
    path_dir = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/elevationQuadHash/"
    )
    tot = len(os.listdir(path_dir))
    count = 0
    for q in os.listdir(path_dir):
        if len(os.listdir(path_dir + q)) == 0:
            count += 1
            os.rmdir(path_dir + q)
    print("No data in: ", count, "/", tot, "quadhashes")


if __name__ == "__main__":
    chop_in_quadhash()
    remove_empty_folders()
    print(pyproj.datadir.get_data_dir())

