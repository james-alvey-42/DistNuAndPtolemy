import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad
from scipy.special import zeta, erf
from joblib import Parallel, delayed

points, weights = np.polynomial.legendre.leggauss(1000)

def integrator(f, a, b):
    sub = (b - a) / 2.
    add = (b + a) / 2.

    if sub == 0:
        return 0.

    return sub * np.dot(f(sub * points + add), weights)

def masses(mlightest, order='normal'):
    if order == 'normal':
        m1 = mlightest
        m2 = np.sqrt(m1**2 + 7.55 * 1e-5)
        m3 = np.sqrt(m1**2 + 2.50 * 1e-3)
    else:
        m3 = mlightest
        m1 = np.sqrt(m3**2 + 2.42 * 1e-3)
        m2 = np.sqrt(m1**2 + 7.55 * 1e-5)
    return m1, m2, m3

DENSITY_FACTOR = 1394359034.3850107 / (8 * np.pi * 6.67408 * 1e-11 / 3) # Class density in Mpc^-2 to Msun / Mpc^-3
H = 0.6732117
G = 6.67408 * 1e-11 * 6.76964163871683e-38 # Mpc^3 / (Msun s^2)
OMEGA_M0 = 0.31580709731
OMEGA_L0 = 1 - OMEGA_M0
RHO_CRIT0 = DENSITY_FACTOR * 5.042685747345e-08 # Msun / Mpc^-3
RHO_M0 = OMEGA_M0 * RHO_CRIT0
RS0 = 0.0199 # rs, Mvir and eta from Tab. 1 of 1910.13388 
MVIR = 2.03 * 1e12 # Msun
ETA = 1.
MSUNKPC3_TO_GEVCM3 = 26330276.95609824

class_file = '../data/cosmology.dat'
answer = input('[nu_clustering.py] Is class file located here: {}? (y/n) '.format(class_file))
if answer.lower()[0] == 'y':
	class_data = np.loadtxt(class_file)
else:
	class_file = input('[nu_clustering.py] Insert file path: ')
	class_data = np.loadtxt(class_file)

a, t = 1 / (1 + class_data[:, 0]), 3.154 * 1e16 * class_data[:, 1]
s_arr = np.array([])
for idx in range(1, len(a)):
	ss = simps(a[:idx]**(-2), t[:idx])
	s_arr = np.append(s_arr, ss)
a_arr, s_arr = a[1:], s_arr
start = 4200
print('zmax = {}'.format(1/a_arr[start] - 1))

def rho_crit(z):
	return RHO_CRIT0 * (OMEGA_M0 * (1 + z)**3 + OMEGA_L0)

def Omega_m(z):
	return RHO_M0 * (1 + z)**3 * np.power(rho_crit(z), -1.0)

def DeltaVir(z):
	return 18 * np.pi**2 + 82 * (Omega_m(z) - 1) - 39 * (Omega_m(z) - 1)**2

def a(z):
	return 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)

def b(z):
	return -0.097 + 0.024 * z

def cvir_avg(z, Mvir=MVIR):
	return 10**(a(z) + b(z) * np.log10(Mvir/(1e12 * H**(-1))))

def rvir(z, Mvir=MVIR):
	return np.power((4 * np.pi * DeltaVir(z) * rho_crit(z)) / (3 * (1 + z)**3 * Mvir), -1./3.)

def cvir(z, Mvir=MVIR):
	beta = rvir(z=0, Mvir=Mvir) / (RS0 * cvir_avg(z=0, Mvir=Mvir))
	return beta * cvir_avg(z)

def rs(z, Mvir=MVIR):
	return rvir(z, Mvir) / cvir(z)

@np.vectorize
def N(z, Mvir=MVIR, eta=ETA):
	integrand = lambda x : 4 * np.pi * x**2 * np.power((x/rs(z, Mvir))**eta * (1 + (x/rs(z, Mvir)))**(3 - eta), -1.0)
	integral = quad(integrand, a=1e-3, b=rvir(z, Mvir))[0]
	return Mvir * (1 + z)**3 * np.power(integral, -1.0)

zarr = 1/a_arr[start:] - 1
Narr = N(zarr)
rsarr = rs(zarr)
N_func = interp1d(zarr, Narr, kind='linear', fill_value='extrapolate')
rs_func = interp1d(zarr, rsarr, kind='linear', fill_value='extrapolate')
def nfw_halo(r, z, Mvir=MVIR, eta=ETA):
	return N_func(z) * np.power((r/rs_func(z))**eta * (1 + (r/rs_func(z)))**(3 - eta), -1.0)

def delta_m(r, z):
	return nfw_halo(r, z) / (RHO_M0 * (1 + z)**3)


def fourier(k, r, delta):
	integrand_arr = 4 * np.pi * r * np.sin(k * r) * delta / k
	return simps(integrand_arr, r)

def ifourier(r, k, delta_f):
	integrand_arr = (1 / (2 * np.pi**2)) * k * np.sin(k * r) * delta_f / r
	return simps(integrand_arr, k)

def F(q, nmax=1000):
	n_arr = np.arange(1, nmax + 1)
	Tnu = 8.372266346757365e-19 * 1.95 # eV Mpc / s
	prefactor = 4 / (3 * zeta(3))
	sum_arr = (-1 * (-1.0)**n_arr) * n_arr / (n_arr**2 + q**2 * Tnu**2)**2
	return prefactor * np.sum(sum_arr)

def f0(p, Tf=1.0):
	Tnu = 8.372266346757365e-19 * 1.95
	return np.power(1 + np.exp(p/(Tnu/Tf)), -1.0)

def f0_gaussian(p, ystar, sigma, Neff):
	amp = Neff * np.pi**4 * 7. * pow(4./11., 4./3.) / 45. / 8. / (sigma**2 * (ystar**2 + 2.* sigma**2) * np.exp(-ystar**2/2./sigma**2) + np.sqrt(np.pi/2.)*ystar*sigma*(ystar**2+3.*sigma**2)*(1.+erf(ystar/np.sqrt(2.)/sigma)));
	Tnu = 8.372266346757365e-19 * 1.95
	p_over_T = p / Tnu
	return amp * np.exp(-(p_over_T**2 - ystar**2)/(2 * sigma**2))

def run_FD(ml, Tf=1.0, show_plot=False):
	q_arr = np.geomspace(1e15, 1e21, 10000)
	p_arr = 1/q_arr[::-1]
	f0_arr = f0(p_arr, Tf=Tf)
	n0 = simps(4 * np.pi * p_arr**2 * f0_arr, p_arr)
	F_arr1 = np.array([])
	F_arr2 = np.array([])
	for q in q_arr:
		F_arr1 = np.append(F_arr1, F(q))
		F_arr2 = np.append(F_arr2, 1/n0 * fourier(q, p_arr, f0_arr))
	F_func = interp1d(q_arr, F_arr2, kind='cubic')
	
	Rmin = 1e-3
	Rmax = 100
	r_arr = np.geomspace(Rmin, Rmax, 100)
	logMvir = np.log10(MVIR)
	# Mvir = (0.7 / H) * 10**logMvir

	print('[nu_clustering.py] Starting mlight = {} eV, Tnu/Tnu0 = {}'.format(ml, Tf))
	try:
		deltanu_f = np.loadtxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(logMvir, ml, Tf))
		run = False
	except OSError:
		run = True
	# run_ans = input('[nu_clustering.py] Check for fourier file (y/n)? ')
	# if run_ans.lower()[0] == 'n':
	# 	run = True

	run = True
	k_arr = 1/r_arr[::-1]
	if run:
		deltanu_f = np.array([])
		for idx, k in enumerate(k_arr):
			print('[nu_clustering.py] {} out of {}'.format(idx + 1, len(k_arr)), end='\r')
			integrand_arr = np.array([])
			for a, s in zip(a_arr[start:], s_arr[start:]):
				diffs = s_arr[-1] - s
				try:
					q = k * diffs / ml
					F_factor = F_func(q)
				except ValueError:
					if q < 1e15:
						F_factor = 1.0
					else:
						F_factor = 0.0
				delta = fourier(k, r_arr, delta_m(r_arr, z=(1/a) - 1))
				integrand_arr = np.append(integrand_arr, a * delta * diffs * F_factor)
			deltanu = 4 * np.pi * G * RHO_M0 * simps(integrand_arr, s_arr[start:])
			deltanu_f = np.append(deltanu_f, deltanu)
		np.savetxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(logMvir, ml, Tf), deltanu_f)
	deltanu_f = np.loadtxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(np.log10(MVIR), ml, Tf))

	deltanu_arr = np.array([])
	for r in r_arr:
		deltanu_arr = np.append(deltanu_arr, ifourier(r, k_arr, deltanu_f))

	np.savetxt('clustering_data/deltanu_real_Mvir{}_mlight{}.txt'.format(logMvir, ml), deltanu_arr)

	if show_plot:
		plt.loglog(r_arr / H**(-1), deltanu_arr + 1)
		plt.xlabel(r'$r\,\mathrm{[h}^{-1}\,\mathrm{Mpc]}$')
		plt.ylabel(r'$n_\nu / \bar{n}_\nu$')
		plt.xlim(1e-3, 1e1)
		plt.show()
	delta = ifourier(8.0/1e3, k_arr, deltanu_f)
	print('[nu_clustering.py] Finished mlight = {} eV, Tnu/Tnu0 = {}: delta = {}'.format(ml, Tf, delta))
	return delta

if __name__ == '__main__':
	pts = input("[nu_clustering.py] num(pts): ")
	filename = input('[nu_clustering.py] Insert filename: ')
	filename = '../data/' + filename
	lcdm = input('[nu_clustering.py] LCDM (y/n)? ')
	print('[nu_clustering.py] Save file: {}'.format(filename))
	mlight_arr = np.geomspace(1e-3 * 10.0, 1e-3 * 1000.0, int(pts))
	if lcdm.lower()[0] == 'y':
		Tf_arr = np.ones(len(mlight_arr))
	else:
		Tf_arr = ((mlight_arr)/(0.08/3.))**(1./3.)

	result = np.array([])
	for idx, Tfact in enumerate(Tf_arr):
		to_add = run_FD(ml=mlight_arr[idx], Tf=Tfact)
		result = np.append(result, to_add)

	to_save = np.vstack([mlight_arr, Tf_arr, result]).T
	np.savetxt(filename, to_save, header='mlight/eV Tf delta')
