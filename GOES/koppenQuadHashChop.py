import os
import geopandas as gpd
import numpy as np
from osgeo import gdal


def chop_in_quadhash(output_base_path, koppen_tif_path):
    root_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/states_quads(09)/"
    )

    quadhash_dir = next(
        d
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and d.startswith("quadshape_9_")
    )

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, "quadhash.shp"))

    os.makedirs(output_base_path, exist_ok=True)

    count = 0
    total = len(quadhashes)

    for ind, row in quadhashes.iterrows():
        poly, quadkey = row["geometry"], row["Quadkey"]

        count += 1
        print(f"Splitting: {count} / {total} for quadkey: {quadkey}")

        quadkey_folder = os.path.join(output_base_path, quadkey)
        os.makedirs(quadkey_folder, exist_ok=True)

        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        output_tif_path = os.path.join(quadkey_folder, "koppen.tif")

        gdal.Translate(output_tif_path, koppen_tif_path, projWin=window)

        if os.path.exists(output_tif_path):
            x = gdal.Open(output_tif_path).ReadAsArray()
            if np.min(x) == np.max(x) == 0:
                os.remove(output_tif_path)  
                print(f"Removed empty file: {output_tif_path}")
            else:
                print(f"Created: {output_tif_path}")
        else:
            print(f"Failed to create: {output_tif_path}")


if __name__ == "__main__":
    output_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppenQuadHash/"
    )

    koppen_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/koppen_geiger_tif/1991_2020/koppen_geiger_0p1.tif"

    chop_in_quadhash(output_base_path, koppen_tif_path)
