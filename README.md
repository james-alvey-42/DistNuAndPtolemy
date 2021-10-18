# Neutrino Distributions and Ptolemy
Repository for calculating cosmological obsevables and performing MCMC analysis for neutrinos with a distribution function different to a thermal Fermi-Dirac one. Also contains code to compute event rates and sensitivity at a Ptolemy-like experiment for neutrions with a given distribution function and mass.

## File Structure

**/class_files/**
  
*background.c* - modified version of the corresponding source file in the class cosmological code. Contains implementation of a FD, as well as a Gaussian distribution (with an optional additional FD component) as specified by the ncdm_parameters (Neff, ystar, sigma, T_FD, gauss).

**/data/**

*cmb-s4_sensitivity.txt* - optimal sensitivity of a CMB-S4 like experiment to modified distribution functions, as compared to a fiducial Fermi-Dirac distribution with the same non-relativistic/relativistic energy density in neutrinos

*planck_sensitivity.txt* - optimal sensitivity of a Planck like experiment to modified distribution functions, as compared to a fiducial Fermi-Dirac distribution with the same non-relativistic/relativistic energy density in neutrinos

**/plotting/**

*dist_sensitivity.py* - plotting for the sensitivity of CMB-S4/Planck like experiments to modified distribution functions

**/montepython_files/**

*run_dist_sensitivity.py* - python wrapper for varying the fiducial model in Planck/CMB-S4 sensitivity analysis (should check the correct fiducial file specification in fake_planck_bluebook.data and similarly for CMB-S4). Modifies the relevant .param files in /nudist_forecast/ before running the full analysis using sensitivity.sh

*run_dist_sensitivity.sh* - removes the current fiducial data file to prepare for a new run, then computes the likelihood for the relevant 

*data.py* - modified montepython/data.py file to include reading in log(ystar), log(sigmastar), Neffncdm, gauss, and T_FD parameters

*default_nu_dist.conf* - configuration file pointing to the correct class installation where background.c has been modified

*/nu_param_files/* - param files for full MCMC runs including all example distributions, as well as complete gaussian+DR analysis

*/nu_run_files/* - bash files to run all cases in /nu_param_files/

*/nudist_forecast/* - param files for computing the fiducial sensitivity of CMB-S4 and Planck like experiments
