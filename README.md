# Rainfall Erosivity Estimation

This repository contains input data, Python functions for estimating rainfall erosivity from individual storm events, and a Jupyter notebook.  
To use the repository, download all files and place the input data, notebook, and function scripts into a single folder.  

After installing the required packages, you can generate the time series of monthly rainfall erosivity for each site by opening the notebook and selecting **“Run All”**.  


## Required Package

This project requires Python 3.8+ and the following packages:

- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  

### Install with pip
```bash
pip install pandas numpy
```

## Input Data Structure

The input data are organized in the folder **`Rain_Data_High_Quality`** with the following hierarchy:  

- **`<stid>`**: Oklahoma Mesonet site ID  
- **`<year>`**: Year of observation  
- **`<stid>_<yearmonth>.csv`**: CSV file containing three variables:  
  - **`stid`**: Site ID  
  - **`time`**: Timestamp in the format `YYYY-MM-DD HH:MM:SS`  
  - **`rain`**: Cumulative rainfall (mm, SI units)

The input data must be stored following this hierarchy.  

## Functions for Erosivity Estimation



## Jupyter Notebook

The Jupyter notebook **`Reproduction.ipynb`** demonstrates how to reproduce the time series of monthly rainfall erosivity from 5-minute interval rainfall records in **`Rain_Data_High_Quality`** using the provided functions.  

the required 


