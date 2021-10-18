#!/bin/bash

rm data/fake_planck_bluebook_fiducial.dat
rm -r MFD; rm -r HEDR; rm -r HELT; rm -r HE; rm -r LTDR;
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/massive_FD.param --output MFD -f 0
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/massive_FD.param --output MFD -f 0
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/highEnergy_plus_DR.param --output HEDR -f 0
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/highEnergy_plus_lowT.param --output HELT -f 0
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/highEnergy.param --output HE -f 0
python2 montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param --output LTDR -f 0

