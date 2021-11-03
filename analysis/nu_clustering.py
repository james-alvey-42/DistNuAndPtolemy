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
DELTA_VIR = 200
RHO_CRIT0 = 5.042685747345e-08 # Mpc^-2
RHO_M0 = DENSITY_FACTOR * OMEGA_M0 * RHO_CRIT0

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


def concentration(z, Mvir):
	return 9 / (1 + z) * np.power(Mvir / (1.5 * 1e13 / H), -0.13)

def rs(z, Mvir):
	return 9.51 * 1e-1 * (1/concentration(z, Mvir)) * OMEGA_M0**(-1/3) * DELTA_VIR**(-1/3) * np.power(Mvir / (1e12 * H**(-1)), 1/3) * H**(-1)

def nfw_halo(r, z, Mvir):
	rs_nfw = rs(z, Mvir)
	c_nfw = concentration(z, Mvir)
	rho_nfw = Mvir / (4 * np.pi * (1/(1 + z))**3 * rs_nfw**3 * (np.log(1 + c_nfw) - (c_nfw / (1 + c_nfw))))
	return rho_nfw * np.power((r / rs_nfw) * (1 + (r / rs_nfw))**2, -1.0)

def delta_m(r, z, Mvir):
	return nfw_halo(r, z, Mvir) / (RHO_M0 * (1 + z)**3)

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
	logMvir = 12
	Mvir = (0.7 / H) * 10**logMvir

	print('[nu_clustering.py] Starting mlight = {} eV, Tnu/Tnu0 = {}'.format(ml, Tf))
	try:
		deltanu_f = np.loadtxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(logMvir, ml, Tf))
		run = False
	except OSError:
		run = True

	k_arr = 1/r_arr[::-1]
	if run:
		deltanu_f = np.array([])
		for idx, k in enumerate(k_arr):
			print('[nu_clustering.py] {} out of {}'.format(idx + 1, len(k_arr)), end='\r')
			integrand_arr = np.array([])
			for a, s in zip(a_arr, s_arr):
				diffs = s_arr[-1] - s
				try:
					q = k * diffs / ml
					F_factor = F_func(q)
				except ValueError:
					if q < 1e15:
						F_factor = 1.0
					else:
						F_factor = 0.0
				delta = fourier(k, r_arr, delta_m(r_arr, z=(1/a) - 1, Mvir=Mvir))
				integrand_arr = np.append(integrand_arr, a * delta * diffs * F_factor)
			deltanu = 4 * np.pi * G * RHO_M0 * simps(integrand_arr, s_arr)
			deltanu_f = np.append(deltanu_f, deltanu)
		np.savetxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(logMvir, ml, Tf), deltanu_f)
	deltanu_f = np.loadtxt('clustering_data/deltanu_fourier_Mvir{}_mlight{}_Tf{}.txt'.format(logMvir, ml, Tf))

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
	filename = 'clustering_data/lcdm_clustering_[t]{}.txt'.format(pts)
	print('[nu_clustering.py] Save file: {}'.format(filename))
	#mlight_arr = np.geomspace(1e-3 * 10.0, 1e-3 * 1000.0, int(mlight_pts))
	Tf_arr = np.linspace(0.3, 1.3, int(pts))

	result = np.array([])
	for Tfact in Tf_arr:
		to_add = run_FD(ml=1.0, Tf=Tfact)
		result = np.append(result, to_add)

	to_save = np.vstack([Tf_arr, result]).T
	np.savetxt(filename, to_save, header='[mlight=1.0 eV] Tf delta')
