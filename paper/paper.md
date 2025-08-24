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
   
3. **`erosive_storms.py`**: This function excludes storm events with total rainfall < 12.7 mm. The remaining events are treated as erosive storms and are used to estimate rainfall erosivity [USDA-ARS (2013)](https://www.ars.usda.gov/ARSUserFiles/60600505/rusle/rusle2_science_doc.pdf).
   
4. **`rainfall_energy.py`**: This function calculates unit rainfall energy using the kinetic energy equation and then derives the rainfall energy in each time interval as described in [USDA-ARS (2013)](https://www.ars.usda.gov/ARSUserFiles/60600505/rusle/rusle2_science_doc.pdf).

5. **`max_30_min_rainfall.py`**: This function identifies the maximum rainfall amount within any consecutive 30-minute period using a rolling window method.
   
6. **`rainfall_erosivity.py`**: This function calculates the **kinetic energy of a storm (E)** as the sum of rainfall energy across all time intervals.  
  It then computes the **maximum 30-minute intensity (I₃₀)** from the maximum 30-minute rainfall amount, converted to mm/hr.  
  Finally, the storm erosivity is determined as **E × I₃₀**.
   
7. **`monthly_erosivity.py`**: This function aggregates rainfall erosivity from individual storms to obtain monthly total erosivity for each site.


# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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
