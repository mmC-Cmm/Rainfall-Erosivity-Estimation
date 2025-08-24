---
title: 'Rainfall Erosivity Estimation Using Python'
tags:
  - Python
  - rainfall erosivity
  - water erosion
  - RUSLE2
    
authors:
  - name: Mengting Chen
    orcid: 0000-0002-6971-5384
    affiliation: "1" 
affiliations:
 - name: School of Civil and Environmental Engineering, Oklahoma State University, United States
   index: 1

date: 23 August 2025
bibliography: paper.bib
---

# Summary

Rainfall erosivity is a key parameter in the Universal Soil Loss Equation (USLE), used to predict soil erosion caused by water. It was developed to quantify the combined effects of rainfall and runoff on soil loss [@Wischmeier & Smith:1978].  

In this study, rainfall erosivity was computed using the kinetic energy equation based on 5-minute interval rainfall records. Python functions developed in this project are designed to estimate rainfall erosivity from individual storm events. The description of the functions are below:

1. **`identify_storms.py`**: This function removes missing data and identifies individual storms.  

2. **`process_intervals.py`**: Since Mesonet’s 5-minute data are cumulative rainfall, this function computes the rainfall amount (mm) and rainfall intensity (mm/hr) for each time interval.
   
3. **`erosive_storms.py`**: This function excludes storm events with total rainfall < 12.7 mm. The remaining events are treated as erosive storms and are used to estimate rainfall erosivity [@USDA-ARS:2013].
  
4. **`rainfall_energy.py`**: This function calculates unit rainfall energy using the kinetic energy equation and then derives the rainfall energy in each time interval as described in '@USDA-ARS:2013'.

5. **`max_30_min_rainfall.py`**: This function identifies the maximum rainfall amount within any consecutive 30-minute period using a rolling window method.
   
6. **`rainfall_erosivity.py`**: This function calculates the **kinetic energy of a storm (E)** as the sum of rainfall energy across all time intervals.  
  It then computes the **maximum 30-minute intensity (I₃₀)** from the maximum 30-minute rainfall amount, converted to mm/hr.  
  Finally, the storm erosivity is determined as **E × I₃₀**.
   
7. **`monthly_erosivity.py`**: This function aggregates rainfall erosivity from individual storms to obtain monthly total erosivity for each site.


# Statement of need

The Python functions developed in this project provide researchers with a framework for estimating rainfall erosivity. These functions automate the processing of high-resolution rainfall data, identify storm events, and compute storm-based erosivity using the kinetic energy equation. The outputs can be aggregated into monthly and annual time series, which are directly applicable for analyzing historical trends, estimating soil loss with USLE-based models, and evaluating temporal variability. By using these tools, researchers can efficiently generate erosivity datasets from gauge observations, providing reliable reference values for calibration, validation, and training when applying regression approaches or machine learning models to predict rainfall erosivity.

# References
