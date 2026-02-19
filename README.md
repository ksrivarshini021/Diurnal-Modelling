# Diurnal-Modelling-using-Vision-Transformers
Applying Vision Transformers to model diurnal patterns in Land Surface Temperature (LST) using GOES-R satellite imagery

# Introduction
This study introduces DAYVIEW, a spatiotemporal deep learning framework designed to reconstruct full diurnal cycles of Land Surface Temperature from a single satellite observation, regardless of acquisition time. The methodology involves using hourly products from GOES-R satellite series over the contiguous United States and integrates ancillary information such as climatic zones and elevation.
 Built on a Masked Autoencoder strategy that has a Vision transformer encoder blocks within the model, DAYVIEW directly addresses three main challenges
1. Estimating Diurnal cycles from sparse observation
2. Considering environmental  context through ancillary data to considering the fluctuation
3. Extending predictions accurately across continental scale

# Background 
Diurnal cycles drive 24-hour fluctuations in key environment processes. Among these, the diurnal cycle of Land Surface Temperature(LST) is a critical Essential Climate variable driving 
Energy Balance: Governs surface radiation and heat exchange
Hydrology: Influences evapotranspiration and soil moisture
Ecology: Affects vegetation function and ecosystem dynamics
Thermal dynamics: Determines peak heat exposure

Challenges in capturing complete diurnal LST cycles at scale
In-situ stations: Provide high-frequency data but are spatially sparse and cannot capture large-scale variability
Satellites: offer broad spatial coverage and frequent observations but suffer from cloud occlusions, creating gaps in time series needed for full diurnal reconstruction.
Comparison to other satellite sensors:
	1. LandSat: 5-6 days revisit
	2. VIIRS: once every day
Result: Hard to reconstruct continuous 24-hours LST behavior over large areas.







   
