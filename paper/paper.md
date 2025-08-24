---
title: 'Rainfall erosivity estimation in Python'
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


# Summary

Rainfall erosivity is a key parameter in the Universal Soil Loss Equation (USLE), used to predict soil erosion caused by water. It was developed to quantify the combined effects of rainfall and runoff on soil loss. In this study, rainfall erosivity was computed using the kinetic energy equation based on 5-minute interval rainfall records. Python functions developed in this project is to estimate rainfall erosivity from individual storm events. The description of the functions are below:
1. **`identify_storms.py`**: This function removes missing data and identifies individual storms.  

2. **`process_intervals.py`**: Since Mesonet’s 5-minute data are cumulative rainfall, this function computes the rainfall amount (mm) and rainfall intensity (mm/hr) for each time interval.
   
3. **`erosive_storms.py`**: This function excludes storm events with total rainfall < 12.7 mm. The remaining events are treated as erosive storms and are used to estimate rainfall erosivity [@USDA-ARS:2013]
  
4. **`rainfall_energy.py`**: This function calculates unit rainfall energy using the kinetic energy equation and then derives the rainfall energy in each time interval as described in '@USDA-ARS:2013'.

5. **`max_30_min_rainfall.py`**: This function identifies the maximum rainfall amount within any consecutive 30-minute period using a rolling window method.
   
6. **`rainfall_erosivity.py`**: This function calculates the **kinetic energy of a storm (E)** as the sum of rainfall energy across all time intervals.  
  It then computes the **maximum 30-minute intensity (I₃₀)** from the maximum 30-minute rainfall amount, converted to mm/hr.  
  Finally, the storm erosivity is determined as **E × I₃₀**.
   
7. **`monthly_erosivity.py`**: This function aggregates rainfall erosivity from individual storms to obtain monthly total erosivity for each site.


# Statement of need

The time series of monthly erosivity datasets can be generated through those python functions, which directly used to analyze rainfall erosivity trends over their historical record lengths. The understanding of trend pattern in rainfall erosivity is essential for researchers interested in the predicting future erosivity. From the monthly time series, long-term average monthly and annual erosivity can be derived to support soil loss estimation in Oklahoma, thereby informing local soil and water conservation practices. Interannual variations in erosivity can also guide decision-makers in identifying vulnerable seasons. Moreover, monthly erosivity records provide critical input for deriving land cover and management factors in the USLE-based models7,8. In addition, gauge-based datasets serve as reference data for evaluating the estimates from regression approaches and machine learning models. 


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Acknowledgements

The author acknowledge Oklahoma Mesonet provides rainfall data for this project.

# References
