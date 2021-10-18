# Cosmic Neutrino Distribution Function
Repository for calculating cosmological obsevables and performing MCMC analysis for neutrinos with a distribution function different to a thermal Fermi-Dirac one. Also contains code to compute event rates and sensitivity at a Ptolemy-like experiment for neutrions with a given distribution function and mass.

## File Structure

**/class_files/**
  
*background.c* - modified version of the corresponding source file in the class cosmological code. Contains implementation of a FD, as well as a Gaussian distribution (with an optional additional FD component) as specified by the ncdm_parameters (Neff, ystar, sigma, T_FD, gauss).
