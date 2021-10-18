#!/bin/bash

screen -S rrun1 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S rrun2 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S rrun3 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S rrun4 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S rrun5 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S rrun6 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
