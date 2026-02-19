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
        min_x = min(coord[0] for coord in bounds)  # min longitude (easting)
        max_x = max(coord[0] for coord in bounds)  # max longitude (easting)
        min_y = min(coord[1] for coord in bounds)  # min latitude (northing)
        max_y = max(coord[1] for coord in bounds)  # max latitude (northing)

        # Construct the window as (min_x, min_y, max_x, max_y)
        window = (min_x, min_y, max_x, max_y)

        print(f"Computed window: {window}")

        if window[2] > window[0] and window[3] > window[1]:
            output_tif_path = os.path.join(quadkey_folder, "landCover.tif")

            src_ds = gdal.Open(koppen_tif_path)
            if not src_ds:
                print(f"Failed to open the source TIFF: {koppen_tif_path}")
                return

            src_transform = src_ds.GetGeoTransform()

            src_win_x_offset = int((min_x - src_transform[0]) / src_transform[1])
            src_win_y_offset = int((max_y - src_transform[3]) / abs(src_transform[5]))

            src_win_width = int((max_x - min_x) / src_transform[1])
            src_win_height = int((min_y - max_y) / abs(src_transform[5]))

            print(f"Pixel offsets - X: {src_win_x_offset}, Y: {src_win_y_offset}")
            print(f"Window size - Width: {src_win_width}, Height: {src_win_height}")

            if src_win_width > 0 and src_win_height > 0:
                try:
                    gdal.Translate(
                        output_tif_path,
                        koppen_tif_path,
                        srcWin=[
                            src_win_x_offset,
                            src_win_y_offset,
                            src_win_width,
                            src_win_height,
                        ],
                    )

                    if os.path.exists(output_tif_path):
                        x = gdal.Open(output_tif_path).ReadAsArray()
                        if np.min(x) == np.max(x) == 0:
                            os.remove(output_tif_path)
                            print(f"Removed empty file: {output_tif_path}")
                        else:
                            print(f"Created: {output_tif_path}")
                    else:
                        print(f"Failed to create: {output_tif_path}")
                except Exception as e:
                    print(f"Error while processing quadkey {quadkey}: {str(e)}")
            else:
                print(
                    f"Computed window has non-positive width or height for quadkey {quadkey}: width={src_win_width}, height={src_win_height}"
                )
        else:
            print(f"Invalid window for quadkey {quadkey}: {window}")

        if not poly.is_valid:
            print(f"Invalid geometry for quadkey {quadkey}: {poly}")


if __name__ == "__main__":
    output_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/landCoverQuadHash/"
    )

    koppen_tif_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/Annual_NLCD_LndChg_2022_CU_C1V0/Annual_NLCD_LndChg_2022_CU_C1V0.tif"

    chop_in_quadhash(output_base_path, koppen_tif_path)
