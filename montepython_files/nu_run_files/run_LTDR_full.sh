#!/bin/bash

screen -S run1 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
sleep 20
screen -S run2 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
sleep 20
screen -S run3 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
sleep 20
screen -S run4 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
sleep 20
screen -S run5 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
sleep 20
screen -S run6 -d -m python montepython/MontePython.py run --conf default_nu_dist.conf --param nudist_forecast/lowEnergy_plus_DR.param -f 1.5 --output LTDR_fullrun --superupdate 20 --covmat covmat/LCDM18.covmat --bestfit bestfit/base2015.bestfit
