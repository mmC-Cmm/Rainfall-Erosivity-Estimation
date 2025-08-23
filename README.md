# Rainfall Erosivity Estimation

This repository contains input data, Jupyter notebooks, and functions for estimating rainfall erosivity from individual storm events.  

## Input Data Structure

The input data are organized in the folder **`Rain_Data_High_Quality`** with the following hierarchy:  

- **`<stid>`**: Oklahoma Mesonet site ID  
- **`<year>`**: Year of observation  
- **`<stid>_<yearmonth>.csv`**: CSV file containing three variables:  
  - **`stid`**: Site ID  
  - **`time`**: Timestamp in the format `YYYY-MM-DD HH:MM:SS`  
  - **`rain`**: Cumulative rainfall (mm, SI units)  


