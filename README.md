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

# Research Questions

To reconstruct diurnal patterns for Land Surface temperature at scale using the scattered sparce data that the GOES-R satellite provides, our study revolves around the following research questions:

‣ RQ-1: How can we design a data-driven model that estimates the full diurnal cycle of land surface temperature when measurements are available only at limited temporal intervals, such as once daily?

‣ RQ-2: How can we integrate ancillary data that are closely related to land surface temperature to improve the reconstruction of its diurnal cycle?

‣ RQ-3: How can we extend the estimation of diurnal land surface temperature cycles to large spatial extents?

# Data Preprocessing

### Primary data 
Hourly LST maps from GOES-R ABI split window bands(14 and 15)

### Ancillary data
DEM (Digital Elevation Model) source : https://www.usgs.gov/

and Köppen climate data source: https://www.gloh2o.org/koppen/

### Coverage
Continuous observations with 1-hour temporal resolution and 2-km spatial resolution for the entire conus. 

### Concatenating dataset -> Quadtree indexing -> Input tile














   
