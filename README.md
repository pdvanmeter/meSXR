# meSXR - Code for ME-SXR calibration, operation, and analysis
Created by Patrick VanMeter  
Uploaded September 2020

### Overview
This repository contains code created during and used throughout my PhD project at the University of Wisconsin-Madison. It has been uploaded here for the purpose of preservation, and to allow others to reference or borrow it as needed. The code is provided as-is will almost certaintly need to be modified to run on a new setup, so you are advised to either copy the contents locally or clone the repository.

The Multi-Energy Soft X-Ray (ME-SXR) system is a plasma diagnostic based on a custom calibration of the PILATUS hybrid photon counting detector. Custom calibrations are used to allow simultaneous use of multiple energy thresholds, along a unique combination of spatial and spectral resolution. This code was developed for the PILATUS3 100K-M unit installed on the Madison Symetric Torus in 2018. This library is composed of two separate modules, described below.

For more information about the diagnostic and its applications, see the following references:  
L. F. Delgado-Aparicio, _et al._ Rev. Sci. Instrum. 89, 10G116 (2018); https://doi.org/10.1063/1.5038798  
P. D. VanMeter, _et al._ Rev. Sci. Instrum. 89, 10G119 (2018); https://doi.org/10.1063/1.5037347  
P. D. VanMeter, _et al._ J. Instrum. 14, C09009 (2019); https://doi.org/10.1088/1748-0221/14/09/C09009
P. D VanMeter, PhD Thesis (2020); https://wippl.wisc.edu/pub_files/theses/thesisVanMeter142_2020.pdf

### Repository contents:
__mesxr__: This moduel contains code related to the calibration, configuration, and operation of the ME-SXR detector.  
__mst_ida__: This is my general purpose library for loading in data, modeling diagnostics, and performing analyses.  
__notebooks__: Jupyter notebooks which archive some of my own analyses and provide a basic tutorial for how to use the library.  
__mesxr_runday.py__: A script which shows how to use (and automate) the mesxr.operations.camera module, which arms and operates the detector, on a runday.  
