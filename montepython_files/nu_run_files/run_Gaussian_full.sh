#!/bin/bash

screen -S run1 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S run2 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S run3 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S run4 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S run5 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
sleep 20
screen -S run6 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param Gaussian_plus_DR_full.param -f 1.5 --output chains_Gaussian_plus_DR_full_2 --superupdate 20 --covmat covmat/chains_Gaussian_plus_DR_full.covmat 
# --bestfit bestfit/chains_Gaussian_plus_DR_full.bestfit
