from goes2go import GOES
import matplotlib.pyplot as plt
import os
import pandas as pd

# # setting downloda path env variable
# download_path = "/s/parsons/b/others/sustain/varsh/Python/GOES"
# os.environ['GOES2GO_DOWNLOAD_PATH'] = download_path

# ds = GOES().latest()

# print("Latest GOES data:")
# print(ds)

# ax = plt.subplot(projection=ds.rgb.crs)
# ax.imshow(ds.rgb.TrueColor(), **ds.rgb.imshow_kwargs)
# ax.coastlines()
# plt.show()


# download ABI multichannel Cloud Moisture Imagery Product
# G = GOES(satellite=18, product="ABI-L2-MCMIP", domain="C")


# G.timerange(start="2022-01-01 00:00", end="2022-01-10 00:00")
# startTime = pd.to_datetime("2022-01-01 00:00")
# endTime = pd.to_datetime("2022-01-07 00:00")

# hourlyRange = pd.date_range(start=startTime, end=endTime, freq="H")

# # for start, end in G.hrRange(start=startTime, end=endTime):
# for currentTime in hourlyRange:
#     try:
#         files = G.timerange(start=currentTime, end=currentTime + pd.Timedelta(hours=1))

#         if len(files) > 0:
#             # ds = G.getFiles(files=[files.iloc[0]])
#             ds = G.get_files(files=files[:1])
#             print(f"download: {ds[0]}")
#         else:
#             print(f"No files available for hour {currentTime}")

#     except Exception as e:
#         print(f"Error during donwload for hour {currentTime}: {e}")

# print("download complete")
