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
- **`<stid>_<yearmonth>.csv`**: CSV file containing 5-minute interval rainfall records with three variables:  
  - **`stid`**: Site ID  
  - **`time`**: Timestamp in the format `YYYY-MM-DD HH:MM:SS`  
  - **`rain`**: Cumulative rainfall (mm, SI units)  

Although the example provided uses Oklahoma Mesonet records, the input data are not limited to this source. Any cumulative rainfall dataset that follows this hierarchy and format will work.  


## Functions for Erosivity Estimation

This repository provides eight Python functions (scripts) that can be used sequentially to estimate rainfall erosivity from raw rainfall data:  

1. **`identify_storms.py`**  
   This function removes missing data and identifies individual storms.  
   - Missing or negative rainfall values are replaced with `0`.  
   - A storm is defined as a period when the rainfall (`rain`) value begins to increase from `0` and continues until the accumulation stops.  

   The output files only contain rainfall records during storm events, organized in the following hierarchy:
    - **`<stid>`**: Oklahoma Mesonet site ID  
    - **`<year>`**: Year of observation  
    - **`<stid>_<yearmonth>.csv`**: Each CSV file corresponds to a single storm and includes 5-minute rainfall data with three variables:
      - **`stid`**: Site ID  
      - **`time`**: Timestamp in the format `YYYY-MM-DD HH:MM:SS`  
      - **`rain`**: Cumulative rainfall (mm, SI units)  
 
2. **`process_intervals.py`**:
   Since Mesonet’s 5-minute data are cumulative rainfall, this function computes the rainfall amount (mm) and rainfall intensity (mm/hr) for each time interval.
   
3. **`erosive_storms.py`**
   This function excludes storm events with total rainfall < 12.7 mm. The remaining events are treated as erosive storms and are used to estimate rainfall erosivity.
   
4. **`rainfall_energy.py`**
   - **`rainfall_energy.py`**  
  This function calculates **rainfall energy** in each time interval using the **kinetic energy equation** as described in USDA-ARS (2013)  
  [RUSLE2 Science Documentation (PDF)](https://www.ars.usda.gov/ARSUserFiles/60600505/rusle/rusle2_science_doc.pdf).

   
9. **`max_30_min_rainfall.py`**  
10. **`rainfall_erosivity.py`**  
11. **`monthly_erosivity.py`**  

## Jupyter Notebook

The Jupyter notebook **`Reproduction.ipynb`** demonstrates how to reproduce the time series of monthly rainfall erosivity from 5-minute interval rainfall records in **`Rain_Data_High_Quality`** using the provided functions.  
The notebook is organized into eight sections, each corresponding to a specific function that can be run step by step. Alternatively, the entire workflow can be executed at once using **“Run All.”**  
Users may define their own input and output paths or use the default settings provided in the notebook.  




