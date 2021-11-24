# Neutrino Distributions and Ptolemy
Repository for calculating cosmological obsevables and performing MCMC analysis for neutrinos with a distribution function different to a thermal Fermi-Dirac one. Also contains code to compute event rates and sensitivity at a Ptolemy-like experiment for neutrions with a given distribution function and mass.

## File Structure

**/analysis/**
  
*ptolemy_analysis.py* - main analysis framework for extracting the Ptolemy sensitivity, can be run directly and asks the user for a choice of exposure, neutrino mass ordering, tritium mass etc. Saves results to a designated file which is readable by plotting utils.

*nu_clustering.py* - main file for computing the clustering factor within the linear regime given a choice of distribution function and neutrino mass

**/class_files/**
  
*background.c* - modified version of the corresponding source file in the class cosmological code. Contains implementation of a FD, as well as a Gaussian distribution (with an optional additional FD component) as specified by the ncdm_parameters (Neff, ystar, sigma, T_FD, gauss).

**/data/**

*cmb-s4_sensitivity.txt* - optimal sensitivity of a CMB-S4 like experiment to modified distribution functions, as compared to a fiducial Fermi-Dirac distribution with the same non-relativistic/relativistic energy density in neutrinos

*planck_sensitivity.txt* - optimal sensitivity of a Planck like experiment to modified distribution functions, as compared to a fiducial Fermi-Dirac distribution with the same non-relativistic/relativistic energy density in neutrinos

*[t]Tyrs_[d]Delta_[m]mT_[o]order_[s]spin_[b]GammaB.txt* - text files containing the sensitivity data for a selection of the fiducial/experimental parameters. All files follow this file naming format, and are read automatically by the corresponding load_ptolemy() function in utils.py.

*lowT.txt, LCDM.txt etc.* - clustering factor as a function of lightest neutrino mass for the various scenarios

**/plotting/**

*dist_sensitivity.py* - plotting for the sensitivity of CMB-S4/Planck like experiments to modified distribution functions

*utils.py* - selection of functions for loading data, plotting curves and adding labels etc.

*main.py* - uses functions in utils.py to create the figures for the paper

*plots/* - folder containing the relevant plots

**/montepython_files/**

*run_dist_sensitivity.py* - python wrapper for varying the fiducial model in Planck/CMB-S4 sensitivity analysis (should check the correct fiducial file specification in fake_planck_bluebook.data and similarly for CMB-S4). Modifies the relevant .param files in /nudist_forecast/ before running the full analysis using sensitivity.sh

*run_dist_sensitivity.sh* - removes the current fiducial data file to prepare for a new run, then computes the likelihood for the relevant 

*data.py* - modified montepython/data.py file to include reading in log(ystar), log(sigmastar), Neffncdm, gauss, and T_FD parameters

*default_nu_dist.conf* - configuration file pointing to the correct class installation where background.c has been modified

*/nu_param_files/* - param files for full MCMC runs including all example distributions, as well as complete gaussian+DR analysis

*/nu_run_files/* - bash files to run all cases in /nu_param_files/

*/nudist_forecast/* - param files for computing the fiducial sensitivity of CMB-S4 and Planck like experiments
