import rasterio


def read_tif_bands(file_path):
    with rasterio.open(file_path) as src:
        num_bands = src.count
        print(f"Number of bands: {num_bands}")

        for band in range(1, num_bands + 1):
            band_data = src.read(band)  
            print(f"Values of Band {band}:")
            print(
                band_data
            ) 


tif_file_path = "/s/parsons/b/others/sustain/varsh/Python/GOES/TifFolder/VNP21A1D_LST_2022001.tif"  
read_tif_bands(tif_file_path)
