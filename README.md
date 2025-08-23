# Rainfall Erosivity Estimation

This repository contains input data, a Jupyter notebook, and functions for estimating rainfall erosivity from individual storm events. The custormer can download this repository save the input data, the jupyter notebook, and all functions in one folder. after install all required package, the time series of monthly rainfall erosivity for each site can be obtained by clicking "Run All".

## Required Package

This project requires Python 3.8+ and the following packages:

- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  

### Install with pip
```bash
pip install pandas numpy


## Input Data Structure

The input data are organized in the folder **`Rain_Data_High_Quality`** with the following hierarchy:  

- **`<stid>`**: Oklahoma Mesonet site ID  
- **`<year>`**: Year of observation  
- **`<stid>_<yearmonth>.csv`**: CSV file containing three variables:  
  - **`stid`**: Site ID  
  - **`time`**: Timestamp in the format `YYYY-MM-DD HH:MM:SS`  
  - **`rain`**: Cumulative rainfall (mm, SI units)

The input data must be stored following this hierarchy.  

## Jupyter Notebook

The Jupyter notebook **`Reproduction_Use.ipynb`** demonstrates how to reproduce the analysis by importing the files from **`Rain_Data_High_Quality`** and applying the provided functions.  
the required 


