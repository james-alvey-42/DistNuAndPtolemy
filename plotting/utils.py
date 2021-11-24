import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.special import zeta, erf
from scipy import stats
import scipy.ndimage
from scipy.interpolate import PchipInterpolator
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.insert(0, '../analysis/')
import ptolemy_analysis as pta

@np.vectorize
def sum_mnu(mlightest_eV, order='normal'):
	if order.lower()[0] == 'n':
		dm21 = 7.42 * 1e-5
		dm31 = 2.514 * 1e-3
		m1 = mlightest_eV
		m2 = np.sqrt(m1**2 + dm21)
		m3 = np.sqrt(m1**2 + dm31)
	else:
		dm21 = 7.42 * 1e-5
		dm32 = 2.497 * 1e-3
		m3 = mlightest_eV
		m2 = np.sqrt(m3**2 + dm32)
		m1 = np.sqrt(m2**2 - dm21)
	return m1 + m2 + m3

def n_ratio(Neff, ystar, sigma):
	prefactor = (7 * np.pi**4 * Neff * (4./11.)**(4/3))/(540 * zeta(3) * 0.71611**4)
	numerator = (ystar * sigma**2 * np.exp(-ystar**2 / (2 * sigma**2))) + (np.sqrt(np.pi/2) * sigma * (ystar**2 + sigma**2) * (1 + erf(ystar/(np.sqrt(2) * sigma))))
	denominator = (sigma**2 * (ystar**2 + 2 * sigma**2) * np.exp(-ystar**2 / (2 * sigma**2))) + (np.sqrt(np.pi/2) * sigma * ystar * (ystar**2 + 3 * sigma**2) * (1 + erf(ystar/(np.sqrt(2) * sigma))))
	return prefactor * numerator / denominator

def n_FD(Tnu=1.95):
	K_TO_CM = 4.366
	return 3 * zeta(3) / (4 * np.pi**2) * Tnu**3 * K_TO_CM**3

def delta_LCDM(mlight_eV, linear=False):
	if linear:
		return 8.1 * mlight_eV**(1.51)
	else:
		return 76.5 * mlight_eV**(2.21)

def load_ptolemy(Tyrs=1.0, Delta=100.0, mT=100.0, Gammab=1e-5, order='normal', spin='Dirac', filename=None, data_dir='../data/'):
	if filename is None:
		filename = '[t]{}_[d]{}_[m]{}_[o]{}_[s]{}_[b]{}.txt'.format(Tyrs, Delta, mT, order.lower()[0], spin.lower()[0], Gammab)
	try:
		data = np.loadtxt(data_dir + filename)
		return data
	except OSError:
		print('[utils.py] (ERROR) Cannot find file:', data_dir + filename)

def plot_ptolemy(data, gauss_filter=1.5, ctr_labels=True, overlay=False):
	mlight, nloc, sensitivity = data[:, 0], data[:, 1], data[:, 2]
	num_m = len(np.unique(mlight))
	num_n = len(np.unique(nloc))
	M, N, S = mlight.reshape(num_n, num_m), nloc.reshape(num_n, num_m), sensitivity.reshape(num_n, num_m)
	
	mask = (S >= 4.0)
	S[mask] = 4.0
	if gauss_filter is not None:
		S = gaussian_filter(S, gauss_filter)
	if not overlay:
		CS = plt.contour(M, N, S, levels=[1, 2, 3], colors='k', linewidths=1.4, zorder=1, alpha=0.85)
		plt.contourf(M, N, S, levels=[1, 2, 3, 4], colors=["#c1e3c9", "#93cea1", "#5aa76d"], vmin=0, vmax=4, zorder=0, alpha=0.85)

		if ctr_labels is not None:
			fmt = {}
			strs = ['68\%', '95\%', '99.7\%']
			for l, s in zip(CS.levels, strs):
				fmt[l] = s
			plt.clabel(CS, CS.levels[:2], inline=True, inline_spacing=35, fmt=fmt, fontsize=12, manual=[(717, 15), (717, 33)])
			plt.clabel(CS, CS.levels[2:], inline=True, inline_spacing=45, fmt=fmt, fontsize=12, manual=[(670, 56)])
		plt.text(250.0, 240.0, 'PTOLEMY', fontsize=18, color='lightgreen')
	else:
		CS = plt.contour(M, N, S, levels=[1, 2, 3], colors='k', linewidths=0.9, zorder=1, alpha=0.85, linestyles='--')
	plt.xscale('log')
	plt.yscale('log')

def add_case_labels(Tyrs=1, Delta=100, mT=100, Gammab=1e-5, order='Normal', spin='Dirac', xmin=12.0):
	plt.text(xmin, 6.2, r'$\mathrm{' + r'{}'.format(order) + r'\ ordering}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 4.7, r'$\mathrm{' + r'{}'.format(spin) + r'}\ \nu$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 3.3, r'$T = ' + r'{}'.format(int(Tyrs)) + r'\, \mathrm{yr}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 2.4, r'$\Delta = ' + r'{}'.format(int(Delta)) + r'\, \mathrm{meV}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.75, r'$M_\mathrm{T} ='  + r'{}'.format(int(mT)) + r'\ \mathrm{g}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.3, r'$\Gamma_\mathrm{b} = 7 \times 10^{'  + r'{}'.format(int(np.log10(Gammab)) - 2) + r'}\ \mathrm{Hz\ eV}^{-1}$', fontsize=12, color='k', rotation=0)

def add_rate_axis(spin='Dirac', labelpad=20, nolabel=False):
	if 'maj' in spin.lower():
		cfactor = 2.0
	else:
		cfactor = 1.0
	
	ax = plt.gca()
	twinax = plt.gca().twinx()
	if not nolabel:
		twinax.set_ylabel(r'$\Gamma_\mathrm{CNB}\,\mathrm{[yr^{-1}\, (100\, g)}^{-1}\mathrm{]}$', rotation=-90, labelpad=labelpad, fontsize=20)
	twinax.set_yscale('log')
	twinax.set_xscale('log')
	twinax.set_ylim(0.0765384247003936 * 1e0 * cfactor, 0.0765384247003936 * 1e3)
	plt.sca(ax)

def set_xy_lims(xmin=10, xmax=1000, ymin=1, ymax=1000):
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)

def add_xy_labels(xlabel=r'$m_\mathrm{lightest}\,\mathrm{[meV]}$', ylabel=r'$n_\nu^\mathrm{loc.}\,\mathrm{[cm}^{-3}\mathrm{]}$', fontsize=20):
	plt.xlabel(xlabel, fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)

HIGHP_LABEL = r'$\mathrm{High-}p_\nu$'
LOWT_LABEL = r'$\mathrm{Low-}T_\nu\mathrm{+DR}$'
LCDM_LABEL = r'$\Lambda\mathrm{CDM}$'

def plot_highp(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	
	plt.plot(mlight_arr, n_cosmo_max_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.text(220.0, 3.4, HIGHP_LABEL, fontsize=14, rotation=-32, color="#3F7BB6")
	# plt.text(220.0, 3.4, r'$\langle p_\nu\rangle \gg 3T$', fontsize=14, rotation=-32, color="#3F7BB6")

	plt.fill_between(mlight_arr, n_cosmo_max_arr, np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", edgecolor="#3F7BB6", linewidth=0.0, zorder=-10, alpha=0.2)
	plt.fill_between(mlight_arr, n_cosmo_max_arr, np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", hatch="---", edgecolor="#d9e5f0", linewidth=0.0, zorder=-9, alpha=1.)

def plot_lowT(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	Tnu_arr = 1.95 * np.power(n_cosmo_max_arr/n_FD(), 1./3.)
	
	# clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering.txt', unpack=True)
	clus_data_mlight, _, clus_data_delta = np.loadtxt('../data/lowT.txt', unpack=True)
	delta_fn = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.text(10.8, 74.0, LOWT_LABEL, fontsize=14, rotation=-18, color="purple")
	# plt.text(10.8, 74.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=14, rotation=-18, color="purple")

	plt.fill_between(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", edgecolor="purple", linewidth=0.0, zorder=-10, alpha=0.2)
	plt.fill_between(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", hatch="|||", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=1.)
	
	plt.text(21.0, 325.0, 'CMB', fontsize=16, color='k')
	plt.text(21.0, 210.0, 'Excluded', fontsize=16, color='k')

def plot_LCDM(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	LCDM_pos = np.argmin(np.abs(sum_mnu_arr - 0.12))
	plt.plot(mlight_arr[:LCDM_pos], (1 + delta_LCDM(1e-3 * mlight_arr[:LCDM_pos])) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1.0)
	plt.plot(mlight_arr[LCDM_pos:], (1 + delta_LCDM(1e-3 * mlight_arr[LCDM_pos:])) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1., linestyle=(1, (3, 2)))
	plt.scatter([mlight_arr[LCDM_pos]], [n_FD() * (1 + delta_LCDM(1e-3 * mlight_arr[LCDM_pos]))], 
			marker='*',
			c='darkgoldenrod',
			alpha=1.0, 
			s=300,
			linewidths=0.1,
			edgecolors='k',
			zorder=9
			)
	# plt.text(11.0, 41.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=14, color="darkgoldenrod")
	plt.text(11.0, 41.0, LCDM_LABEL, fontsize=14, color="darkgoldenrod")

def add_KATRIN(forecast=False):
	if not forecast:
		plt.axvline(200.0, c='#BF4145', zorder=-9, ls=(1, (5, 1)), lw=1.2)
		plt.text(168.0, 1.6, 'KATRIN', fontsize=12, rotation=90, color="#BF4145")
		plt.annotate(s='', xytext=(205.0, 2.8), xy=(350.0, 2.8), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': '#BF4145'}, color='#BF4145')
	else:
		plt.axvline(200.0, c='#BF4145', zorder=-10, ls='-', lw=1.2)
		plt.fill_betweenx([1, 1000], 200.0, 1000.0, facecolor="none", color="none", hatch="xxxx", edgecolor='#f2d9da', linewidth=0.0, zorder=-11, alpha=1.)
		plt.text(170.0, 1.3, 'KATRIN', fontsize=12, rotation=90, color="#BF4145",zorder=10)


def add_0vvb(forecast=False):
	if not forecast:
		plt.axvline(480.0, c='#d37c2d', zorder=-11, lw=1.2)
		plt.text(405.0, 1.7, r'$0\nu\beta\beta$', fontsize=14, rotation=90, color="#d37c2d")
		plt.fill_betweenx([1, 1000], 480.0, 1000.0, facecolor="none", color="none", hatch="xxxx", edgecolor='#edcaab', linewidth=0.0, zorder=-11, alpha=1.)
		plt.annotate(s='', xytext=(490.0, 1.9), xy=(834.0, 1.9), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': '#d37c2d'}, color='#d37c2d')
	else:
		plt.text(110.0/3.0, 1.7, r'$0\nu\beta\beta$', fontsize=14, rotation=90, color="#d37c2d")
		plt.axvline(100.0/3.0, c='#d37c2d', zorder=-11, lw=1.2)
		plt.axvline(400.0/3.0, c='#d37c2d', zorder=-11, lw=1.2)
		plt.fill_betweenx([1, 1000], 100.0/3.0, 400.0/3.0, facecolor="none", color="#d37c2d", edgecolor='#d37c2d', linewidth=0.0, zorder=-11, alpha=0.2)


def plot_ptolemy_summnu(data, order='normal', gauss_filter=1.5, ctr_labels=True):
	mlight, nloc, sensitivity = data[:, 0], data[:, 1], data[:, 2]
	num_m = len(np.unique(mlight))
	num_n = len(np.unique(nloc))
	M, N, S = (1./3.) * 1e3 * sum_mnu(1e-3 * mlight.reshape(num_n, num_m), order=order), nloc.reshape(num_n, num_m), sensitivity.reshape(num_n, num_m)
	
	mask = (S >= 4.0)
	S[mask] = 4.0
	if gauss_filter is not None:
		S = gaussian_filter(S, gauss_filter)

	CS = plt.contour(M, N, S, levels=[1, 2, 3], colors='k', linewidths=1.4, zorder=1, alpha=0.85)
	plt.contourf(M, N, S, levels=[1, 2, 3, 4], colors=["#c1e3c9", "#93cea1", "#5aa76d"], vmin=0, vmax=4, zorder=0, alpha=0.85)

	if ctr_labels is not None:
		fmt = {}
		strs = ['68\%', '95\%', '99.7\%']
		for l, s in zip(CS.levels, strs):
			fmt[l] = s
		plt.clabel(CS, CS.levels[:2], inline=True, inline_spacing=35, fmt=fmt, fontsize=12, manual=[(717, 15), (717, 33)])
		plt.clabel(CS, CS.levels[2:], inline=True, inline_spacing=45, fmt=fmt, fontsize=12, manual=[(670, 56)])
	plt.text(220.0, 240.0, 'PTOLEMY', fontsize=18, color='lightgreen')
	plt.xscale('log')
	plt.yscale('log')

def plot_highp_DESI(order='normal'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 10, 1e3 * 1000, 1000)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.04 / sum_mnu_arr)
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, n_upper_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)

	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, n_upper_arr, facecolor="#3F7BB6", color="#3F7BB6", edgecolor="#d9e5f0", linewidth=0.0, zorder=-9, alpha=0.2)
	# plt.text(70.0, 3.45, r'$\langle p_\nu\rangle \gg 3T$', fontsize=14, rotation=-32, color="#3F7BB6")
	plt.text(70.0, 3.45, HIGHP_LABEL, fontsize=14, rotation=-32, color="#3F7BB6")

def plot_lowT_DESI(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 10, 1e3 * 1000, 1000)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.04 / sum_mnu_arr)
	Tnu_upper_arr = 1.95 * np.power(n_upper_arr/n_FD(), 1./3.)
	Tnu_lower_arr = 1.95 * np.power(n_lower_arr/n_FD(), 1./3.)
	
	# clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering_008.txt', unpack=True)
	clus_data_mlight, _, clus_data_delta = np.loadtxt(data_dir + 'lowT_0.08.txt', unpack=True)
	delta_fn_upper = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	# clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering_004.txt', unpack=True)
	clus_data_mlight, _, clus_data_delta = np.loadtxt(data_dir + 'lowT_0.04.txt', unpack=True)
	delta_fn_lower = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_upper((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_upper_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), (1 + delta_fn_upper((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_upper_arr), facecolor="purple", color="purple", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=0.2)
	# plt.text(40., 21.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=14, rotation=-26, color="purple")
	plt.text(6., 105.0, LOWT_LABEL, fontsize=14, rotation=-34, color="purple")
	plt.text(20.5, 15, 'DESI/EUCLID', fontsize=14, rotation=-34, color='k')

def plot_LCDM_DESI(order='normal'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 40, 1e-3 * 80, 1000)
	LCDM_pos = np.argmin(np.abs(sum_mnu_arr - 0.08))
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_LCDM((1./3.) * sum_mnu_arr)) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1.0)
	plt.scatter([40./3., 80./3.], [n_FD() * (1 + delta_LCDM(0.040/3.)), n_FD() * (1 + delta_LCDM(0.080/3.))], 
			marker='|',
			c='darkgoldenrod',
			alpha=1.0, 
			s=100,
			linewidths=0.1,
			edgecolors='k',
			zorder=9
			)
	sum_mnu_arr = np.geomspace(1e-3 * 15, 1e-3 * 3000, 1000)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_LCDM((1./3.) * sum_mnu_arr)) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-10, ls=(1, (5, 1)), alpha=0.5)
	# plt.text(25., 70.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=14, rotation=10, color="darkgoldenrod")
	plt.text(25., 70.0, LCDM_LABEL, fontsize=14, rotation=10, color="darkgoldenrod")

def plot_highp_DESI_no_detect(order='inverted'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 1, 1e3 * 1000, 1000)
	n_lower_arr = n_FD() * (0.02 / sum_mnu_arr)
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)

	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, np.repeat(1e3, len(sum_mnu_arr)), facecolor="none", color="none", hatch="---", edgecolor="#d9e5f0", linewidth=0.0, zorder=-9, alpha=1.)
	# plt.text(12.0, 9.8, r'$\langle p_\nu\rangle \gg 3T$', fontsize=14, rotation=-34, color="#3F7BB6")
	plt.text(12.0, 9.8, HIGHP_LABEL, fontsize=14, rotation=-34, color="#3F7BB6")

def plot_lowT_DESI_no_detect(order='inverted', data_dir='../data/'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 1, 1e3 * 1000, 1000)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.02 / sum_mnu_arr)
	Tnu_upper_arr = 1.95 * np.power(n_upper_arr/n_FD(), 1./3.)
	Tnu_lower_arr = 1.95 * np.power(n_lower_arr/n_FD(), 1./3.)
	
	clus_data_mlight, _, clus_data_delta = np.loadtxt(data_dir + 'lowT_0.08.txt', unpack=True)
	delta_fn_upper = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	# clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering_004.txt', unpack=True)
	clus_data_mlight, _, clus_data_delta = np.loadtxt(data_dir + 'lowT_0.04.txt', unpack=True)
	delta_fn_lower = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')

	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), np.repeat(1e3, len(sum_mnu_arr)), facecolor="none", color="none", hatch="|||", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=1.)
	# plt.text(11.0, 17.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=14, rotation=-34, color="purple")
	plt.text(11.0, 17.0, LOWT_LABEL, fontsize=14, rotation=-34, color="purple")
	plt.text(6.5, 325.0, 'DESI/EUCLID', fontsize=14, color='k')
	plt.text(6.5, 220.0, 'Excluded', fontsize=14, color='k')

def plot_LCDM_DESI_no_detect(order='inverted'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 1, 1e-3 * 20, 1000)
	LCDM_pos = np.argmin(np.abs(sum_mnu_arr - 0.08))
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_LCDM((1./3.) * sum_mnu_arr)) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1.0)
	plt.scatter([20./3.], [n_FD() * (1 + delta_LCDM(0.020/3.))], 
			marker='|',
			c='darkgoldenrod',
			alpha=1.0, 
			s=100,
			linewidths=0.1,
			edgecolors='k',
			zorder=9
			)
	sum_mnu_arr = np.geomspace(1e-3 * 15, 1e-3 * 3000, 1000)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_LCDM((1./3.) * sum_mnu_arr)) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-10, ls=(1, (5, 1)), alpha=0.5)
	# plt.text(8., 60.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=14, rotation=1, color="darkgoldenrod", zorder=5)
	plt.text(8., 60.0, LCDM_LABEL, fontsize=14, rotation=1, color="darkgoldenrod", zorder=5)

def add_mlightest_zero(order='normal'):
	plt.axvline((1./3.) * 1e3 * sum_mnu(mlightest_eV=0.0, order=order), c='k', zorder=-10, ls=(1, (5, 1)), lw=1.1, alpha=0.4)
	if order.lower()[0] == 'n':
		plt.text(20.5, 250, r'$m_\mathrm{lightest} \geq 0$', rotation=-90, fontsize=12, zorder=10)
		plt.annotate(s='', xytext=(24.5, 500.0), xy=(42.0, 500.0), arrowprops={'arrowstyle': '-|>', 'lw':1.0, 'color': 'k'}, color='k', alpha=0.8)
	else:
		plt.text(34, 250, r'$m_\mathrm{lightest} \geq 0$', rotation=-90, fontsize=12, zorder=10)
		plt.annotate(s='', xytext=(40.5, 500.0), xy=(65.0, 500.0), arrowprops={'arrowstyle': '-|>', 'lw':1.0, 'color': 'k'}, color='k', alpha=0.8)

def plot_highp_simple(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	
	plt.plot(mlight_arr, n_cosmo_max_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.text(220.0, 3.2, HIGHP_LABEL, fontsize=14, rotation=-32, color="#3F7BB6", zorder=-10)
	# plt.text(220.0, 3.2, r'$\langle p_\nu\rangle \gg 3T$', fontsize=14, rotation=-32, color="#3F7BB6", zorder=-10)

def plot_lowT_simple(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	Tnu_arr = 1.95 * np.power(n_cosmo_max_arr/n_FD(), 1./3.)
	
	# clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering.txt', unpack=True)
	clus_data_mlight, _, clus_data_delta = np.loadtxt('../data/lowT.txt', unpack=True)
	delta_fn = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	# plt.text(10.8, 72.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=14, rotation=-18, color="purple", zorder=-10)
	plt.text(10.8, 72.0, LOWT_LABEL, fontsize=14, rotation=-18, color="purple", zorder=-10)


def plot_LCDM_simple(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	LCDM_pos = np.argmin(np.abs(sum_mnu_arr - 0.12))
	plt.plot(mlight_arr[:LCDM_pos], (1 + delta_LCDM(1e-3 * mlight_arr[:LCDM_pos])) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1.0)
	plt.plot(mlight_arr[LCDM_pos:], (1 + delta_LCDM(1e-3 * mlight_arr[LCDM_pos:])) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1., linestyle=(1, (3, 2)))
	plt.scatter([mlight_arr[LCDM_pos]], [n_FD() * (1 + delta_LCDM(1e-3 * mlight_arr[LCDM_pos]))], 
			marker='*',
			c='darkgoldenrod',
			alpha=1.0, 
			s=200,
			linewidths=0.1,
			edgecolors='k',
			zorder=-5
			)
	# plt.text(11.0, 40.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=14, color="darkgoldenrod", zorder=-10)
	plt.text(11.0, 40.0, LCDM_LABEL, fontsize=14, color="darkgoldenrod", zorder=-10)


def plot_background_rates(mlightest, delta, order='normal', n0=56.0, tritium_mass_g=100.0, c='k'):
	m1, m2, m3 = pta.masses(mlightest, order)
	m3H = 2808.921 # MeV
	NT = 1.9972819100287977e+25 * (tritium_mass_g / 100.0)
	Eend_zero = pta.Eend0()
	me = 511 * 1e6 # meV
	Ee_arr = np.linspace(-1000, 3000, 1000) + Eend_zero
	n0 = n0 * 7.685803257085992e-06 # meV^3
	eVyr_factor = 4.794049023619834e+22
	if delta == 0.0:
		beta_arr = pta.dGbdEe(m1, m2, m3, Ee_arr, NT, order)
	else:
		beta_arr = pta.dGtbdE(delta, m1, m2, m3, Ee_arr, NT, order)
	plt.semilogy(Ee_arr - Eend_zero, eVyr_factor * (beta_arr), lw=1.8, c=c, ls=(1, (5, 1)))

def plot_signal_rates(mlightest, delta, order='normal', n0=56.0, tritium_mass_g=100.0, c='k', prefactor=1.0):
	m1, m2, m3 = pta.masses(mlightest, order)
	m3H = 2808.921 # MeV
	NT = 1.9972819100287977e+25 * (tritium_mass_g / 100.0)
	Eend_zero = pta.Eend0()
	me = 511 * 1e6 # meV
	Ee_arr = np.linspace(-1000, 3000, 1000) + Eend_zero
	n0 = n0 * 7.685803257085992e-06 # meV^3
	eVyr_factor = 4.794049023619834e+22
	CNB_arr = pta.dGtCNBdE(delta, m1, m2, m3, Ee_arr, n0, NT, order)
	plt.semilogy(Ee_arr - Eend_zero, eVyr_factor * (CNB_arr), lw=2.2, c=c, ls='-')

def plot_binned_data(Ei_arr, N_arr, delta, filled=True, step_color='k', fill_color='k', fill_alpha=0.2, points=True, zorder=10):
	plt.step(x=Ei_arr - pta.Eend0(), y=N_arr, 
		where='mid', color=step_color, linewidth=1.1, zorder=zorder)
	if filled:
		plt.bar(x=Ei_arr - pta.Eend0(), height=N_arr, 
			align='center', color=fill_color, 
			edgecolor=None, linewidth=0.0, 
			width=Ei_arr[1] - Ei_arr[0], alpha=fill_alpha, zorder=zorder)
	if points:
		plt.scatter(Ei_arr - pta.Eend0(), N_arr, 
			marker='.',
			c=step_color,
			alpha=1.0,
			s=100,
			linewidths=0.2,
			edgecolors='k',zorder=zorder + 1)

def get_event_arrays(Tyrs=1.0, delta=100.0, order='normal', mT=100.0, gamma_b=1e-5, spin='Dirac', mlight=50.0, nloc=10.0):
	NT = 1.9972819100287977e+25 * (mT / 100.)
	if spin.lower()[0] == 'd':
	    cDM_sim = 1.0
	else:
	    cDM_sim = 2.0
	Nb_data = 1.05189 * (gamma_b/1e-5) * (Tyrs / 1.0)
	Ei_arr = np.linspace(pta.Eend0() - 5000, pta.Eend0() + 10000., int(15000/delta))

	N_data_arr = pta.N_total(Ei_arr, Tyrs, delta, mlight, nloc, NT, order=order, DEEnd=0.0, gamma_b=gamma_b, cDM=cDM_sim)
	N_beta_arr = pta.N_beta(Ei_arr, Tyrs, delta, mlight, NT, order, DEEnd=0.0)
	N_CNB_arr = pta.N_CNB(Ei_arr, Tyrs, delta, mlight, nloc, NT, order, DEEnd=0.0, cDM=cDM_sim)
	N_CNB_large_arr = pta.N_CNB(Ei_arr, Tyrs, delta, mlight, 1e2 * nloc, NT, order, DEEnd=0.0, cDM=cDM_sim)
	N_background_arr = N_data_arr - N_beta_arr - N_CNB_arr

	print('[utils.py] Computed events for mlight = {} meV, delta = {} meV'.format(mlight, delta))
	return {'energies': Ei_arr, 'beta': N_beta_arr, 'CNB': N_CNB_arr, 'CNB_large': N_CNB_large_arr, 'background': N_background_arr}

def plot_all_events(arr_dict, unit='eV'):
	Ei_arr = arr_dict['energies']
	N_background_arr = arr_dict['background']
	N_beta_arr = arr_dict['beta']
	N_CNB_arr = arr_dict['CNB']
	N_CNB_large_arr = arr_dict['CNB_large']
	plot_binned_data(Ei_arr, N_background_arr, (Ei_arr[-1] - Ei_arr[-2]), step_color='k', fill_color='k', zorder=0, points=False)	
	plot_binned_data(Ei_arr, N_beta_arr, (Ei_arr[-1] - Ei_arr[-2]), step_color='#3F7BB6', fill_color='#3F7BB6', zorder=2)
	plot_binned_data(Ei_arr, N_CNB_large_arr, (Ei_arr[-1] - Ei_arr[-2]), step_color='darkgoldenrod', fill_color='darkgoldenrod', zorder=4)
	plot_binned_data(Ei_arr, N_CNB_arr, (Ei_arr[-1] - Ei_arr[-2]), step_color='purple', fill_color='purple', zorder=6)
	set_xy_scales()
	set_xy_lims(-2000, 1000, 1e-2, 1e14)
	plt.xticks([-2000, -1000, 0, 1000])
	plt.yticks([1e-2, 1e2, 1e6, 1e10, 1e14])

def plot_snr(arr_dict, unit='eV'):
	Ei_arr = arr_dict['energies']
	N_background_arr = arr_dict['background']
	N_beta_arr = arr_dict['beta']
	N_CNB_arr = arr_dict['CNB']
	N_CNB_large_arr = arr_dict['CNB_large']
	plot_binned_data(Ei_arr, N_CNB_large_arr/np.sqrt(N_beta_arr + N_background_arr), (Ei_arr[-1] - Ei_arr[-2]), step_color='darkgoldenrod', fill_color='darkgoldenrod')
	plot_binned_data(Ei_arr, N_CNB_arr/np.sqrt(N_beta_arr + N_background_arr), (Ei_arr[-1] - Ei_arr[-2]), step_color='purple', fill_color='purple')
	plt.text(-1900, 2.0, r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}} > 1$', c='k', fontsize=12, rotation=0)
	# plt.annotate(s='', xytext=(-300.0, 1.1), xy=(-300.0, 60.0), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': 'k'}, color='k')
	plt.axhline(1.0, lw=1.8)
	set_xy_scales()
	set_xy_lims(-2000, 1000, 1e-2, 1e2)
	plt.xticks([-2000, -1000, 0, 1000])

def add_event_labels(mlight=50.0, delta=100.0, Tyrs=1.0, mT=100.0, gammab=1e-5, side='left', unit='eV'):
	plt.text(-1950, 8.8e9, r'$\beta \mathrm{\ decay\ background}$', c='#3F7BB6', fontsize=12, rotation=-6)
	plt.text(-1950, 0.05, r'$\mathrm{Const.\ background}\,\Gamma_\mathrm{b}$', c='k', fontsize=12, rotation=0)

	# plt.text(-1900, 10**(8.1), r'$\Delta = ' + r'{}'.format(int(delta)) + r'\, \mathrm{meV}$', c='k', fontsize=12, rotation=0)
	# plt.text(-1900, 10**(6.8), r'$m_\mathrm{lightest} = ' + r'{}'.format(int(mlight)) + r'\,\mathrm{meV}$', c='k', fontsize=12, rotation=0)

	# plt.text(-1900, 10**(3.1), r'$T = ' + r'{}'.format(int(Tyrs)) + r'\, \mathrm{yr}$', c='k', fontsize=12, rotation=0)
	# plt.text(-1900, 10**(1.8), r'$m_\mathrm{T} = ' + r'{}'.format(int(mT)) + r'\mathrm{g}$', c='k', fontsize=12, rotation=0)
	# plt.text(-1900, 10**(0.5), r'$\Gamma_\mathrm{b} = 7 \times 10^{' + r'{}'.format(int(np.log10(gammab)) - 2) + r'}\,\mathrm{Hz\ eV}^{-1}$', c='k', fontsize=12, rotation=0)

	if side.lower()[0] == 'l':
		plt.text(-75.0 - 800.0, 10**(5.1), 'CNB Signal', c='k', fontsize=12, rotation=0)
		plt.text(-100.0 - 800.0, 10**(2.5), r'$n_\nu^\mathrm{loc.} = 10 \, \mathrm{cm}^{-3}$', c='purple', fontsize=12, rotation=0)
		plt.text(-112.0 - 800.0, 10**(3.8), r'$n_\nu^\mathrm{loc.} = 10^3 \, \mathrm{cm}^{-3}$', c='darkgoldenrod', fontsize=12, rotation=0)
	elif side.lower()[0] == 'r':
		plt.text(-75.0, 10**(5.1), 'CNB Signal', c='k', fontsize=12, rotation=0)
		plt.text(-100.0, 10**(2.5), r'$n_\nu^\mathrm{loc.} = 10 \, \mathrm{cm}^{-3}$', c='purple', fontsize=12, rotation=0)
		plt.text(-112.0, 10**(3.8), r'$n_\nu^\mathrm{loc.} = 10^3 \, \mathrm{cm}^{-3}$', c='darkgoldenrod', fontsize=12, rotation=0)

def set_xy_scales(xscale='linear', yscale='log'):
	plt.xscale(xscale)
	plt.yscale(yscale)

def cosmo_color(case='LCDM'):
	colors_dict = {
	 'LCDM': '#003f5c',
	 'LEDR': '#58508d',
	 'HE': '#bc5090',
	 'HEDR': '#ff6361',
	 'LTM': '#ffa600'
	 }
	# colors_dict = {
	#  'LCDM': 'purple',
	#  'LEDR': '#306B37',
	#  'HE': 'darkgoldenrod',
	#  'HEDR': '#3F7BB6',
	#  'LTM': '#BF4145'
	# }
	try:
		return colors_dict[case]
	except:
		print('[utils.py] (ERROR) Case not recognised, available cosmo scenarios are:')
		print('[utils.py] \t(1) Key: LCDM - LCDM with FD distribution')
		print('[utils.py] \t(2) Key: LEDR - Low energy neutrinos with additional DR')
		print('[utils.py] \t(3) Key: HE - High Energy neutrinos')
		print('[utils.py] \t(4) Key: HEDR - High Energy neutrinos with additional DR')
		print('[utils.py] \t(5) Key: LTM - Low temperature, but with additional Gaussian component')
		return 'k'

def Amp(Neff, ystar, sigma):
    return Neff * pow(np.pi, 4.) * 7. * pow(4./11., 4./3.) / 45. / 8. / (pow(sigma,2.) * (pow(ystar,2.) + 2.*pow(sigma,2.)) * np.exp(-pow(ystar,2.)/2./pow(sigma,2.)) + np.sqrt(np.pi/2.)*ystar*sigma*(pow(ystar,2.)+3.*pow(sigma,2.))*(1.+erf(ystar/np.sqrt(2.)/sigma)))

def Gaussian_distribution(q, Neff, ystar, sigma):
    return 2 * Amp(Neff, ystar, sigma) / pow(0.71611, 4.) / (pow(2 * np.pi,3)) * np.exp(-pow(q-ystar,2.)/2./pow(sigma,2.))

def FD_distribution(q, Tncdm):
    return 6.0/(pow(2 * np.pi,3))/(np.exp(q/Tncdm)+1.)

def plot_distributions():
	temp_Gauss_func = lambda Amp,ystar,sigma,y : Amp * np.exp(-(y-ystar)**2/(2*sigma**2))
	temp_FD_func    = lambda y : 1./(np.exp(y)+1)
	qarr = np.geomspace(1e-2, 50., 10000)

	Tnu0 = 1.95
	K_TO_CM = 4.366

	plt.plot(qarr, 0.04 * qarr * Tnu0**3 * K_TO_CM**3 * temp_FD_func(qarr) * 6 * qarr**2 / (2 * np.pi**2),
			c=cosmo_color('LCDM'), lw=2.2)
	plt.plot(qarr, 0.04 * qarr * Tnu0**3 * K_TO_CM**3 * temp_Gauss_func(33.8595, 0.1, 0.294218, qarr) * 6 * qarr**2 / (2 * np.pi**2), 
		c=cosmo_color('LEDR'), lw=2.2)
	plt.plot(qarr, 0.40 * qarr * Tnu0**3 * K_TO_CM**3 * temp_Gauss_func(0.0000161608, 30, 4.82113, qarr) * 6 * qarr**2 / (2 * np.pi**2), 
		c=cosmo_color('HE'),lw=2.2)
	plt.plot(qarr, 0.40 * qarr * Tnu0**3 * K_TO_CM**3 * temp_Gauss_func(0.000125406, 3.0, 8.82654, qarr) * 6 * qarr**2 / (2 * np.pi**2), 
		c=cosmo_color('HEDR'), lw=2.2)
	plt.plot(qarr, 0.04 * qarr * Tnu0**3 * K_TO_CM**3 * (temp_Gauss_func(0.0743352, 3.5, 0.508274, qarr) + temp_FD_func(qarr / 0.7003)) * 6 * qarr**2 / (2 * np.pi**2),
		c=cosmo_color('LTM'), lw=2.2)

def plot_energy_evolution(data_dir='../data/distribution_data/', labels=True, nu_only=False):
	file_LCDM  = data_dir + "LCDM_004_background.dat"
	file_LEDR  = data_dir + "Lownus_004_background.dat"
	file_HE  = data_dir + "High_nus_004_background.dat"
	file_HEDR  = data_dir + "High_nus_Mixed_004_background.dat"
	file_LTM  = data_dir + "LowT_High_nus_004_background.dat"

	if nu_only:
		DR_factor = 0.
	else:
		DR_factor = 1.

	rho_LCDM = interp1d(np.loadtxt(file_LCDM)[:,0],np.loadtxt(file_LCDM)[:,11],bounds_error=False,fill_value=0.0,kind='linear')
	rho_LEDR = interp1d(np.loadtxt(file_LEDR)[:,0],(np.loadtxt(file_LEDR)[:,11]+DR_factor * np.loadtxt(file_LEDR)[:,16]),bounds_error=False,fill_value=0.0,kind='linear')
	rho_HE = interp1d(np.loadtxt(file_HE)[:,0],(np.loadtxt(file_HE)[:,11]),bounds_error=False,fill_value=0.0,kind='linear')
	rho_HEDR = interp1d(np.loadtxt(file_HEDR)[:,0],(np.loadtxt(file_HEDR)[:,11]+DR_factor * np.loadtxt(file_HEDR)[:,16]),bounds_error=False,fill_value=0.0,kind='linear')
	rho_LTM = interp1d(np.loadtxt(file_LTM)[:,0],(np.loadtxt(file_LTM)[:,11]),bounds_error=False,fill_value=0.0,kind='linear')

	zvec = np.loadtxt(file_LCDM)[:,0]
	plt.plot(zvec, rho_LCDM(zvec)/rho_LCDM(zvec),c=cosmo_color('LCDM'),lw=2.2)
	plt.plot(zvec, rho_LEDR(zvec)/rho_LCDM(zvec),c=cosmo_color('LEDR'),lw=2.2)
	plt.plot(zvec, rho_HE(zvec)/rho_LCDM(zvec),c=cosmo_color('HE'),lw=2.2)
	plt.plot(zvec, rho_HEDR(zvec)/rho_LCDM(zvec),c=cosmo_color('HEDR'),lw=2.2)
	plt.plot(zvec, rho_LTM(zvec)/rho_LCDM(zvec),c=cosmo_color('LTM'),lw=2.2)

	if labels and not nu_only:
		plt.text(0.35, 0.16, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=0)
		plt.text(0.23, 0.50,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
				transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=61)
		plt.text(0.55, 0.05, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HE'),
			rotation=32)
		plt.text(0.31, 0.35,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=57)
		plt.text(0.075,0.1,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=0)
	elif labels and nu_only:
		plt.text(0.44, 0.92, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=0)
		plt.text(0.84, 0.16,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
				transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=0)
		plt.text(0.53, 0.855, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HE'),
			rotation=7)
		plt.text(0.84, 0.465,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=0)
		plt.text(0.84,0.92,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=0)

def plot_eos(data_dir='../data/distribution_data/', labels=True):
	file_LCDM  = data_dir + "LCDM_004_background.dat"
	file_LEDR  = data_dir + "Lownus_004_background.dat"
	file_HE  = data_dir + "High_nus_004_background.dat"
	file_HEDR  = data_dir + "High_nus_Mixed_004_background.dat"
	file_LTM  = data_dir + "LowT_High_nus_004_background.dat"

	rho_LCDM = interp1d(np.loadtxt(file_LCDM)[:,0],3 * np.loadtxt(file_LCDM)[:,12]/np.loadtxt(file_LCDM)[:,11],bounds_error=False,fill_value=0.0,kind='linear')
	rho_LEDR = interp1d(np.loadtxt(file_LEDR)[:,0],3 * np.loadtxt(file_LEDR)[:,12]/np.loadtxt(file_LEDR)[:,11],bounds_error=False,fill_value=0.0,kind='linear')
	rho_HE = interp1d(np.loadtxt(file_HE)[:,0],3 * np.loadtxt(file_HE)[:,12]/(np.loadtxt(file_HE)[:,11]),bounds_error=False,fill_value=0.0,kind='linear')
	rho_HEDR = interp1d(np.loadtxt(file_HEDR)[:,0],3 * np.loadtxt(file_HEDR)[:,12]/(np.loadtxt(file_HEDR)[:,11]),bounds_error=False,fill_value=0.0,kind='linear')
	rho_LTM = interp1d(np.loadtxt(file_LTM)[:,0],3 * np.loadtxt(file_LTM)[:,12]/(np.loadtxt(file_LTM)[:,11]),bounds_error=False,fill_value=0.0,kind='linear')

	zvec = np.loadtxt(file_LCDM)[:,0]
	plt.plot(zvec, rho_LCDM(zvec),c=cosmo_color('LCDM'),lw=2.2)
	plt.plot(zvec, rho_LEDR(zvec),c=cosmo_color('LEDR'),lw=2.2)
	plt.plot(zvec, rho_HE(zvec),c=cosmo_color('HE'),lw=2.2)
	plt.plot(zvec, rho_HEDR(zvec),c=cosmo_color('HEDR'),lw=2.2)
	plt.plot(zvec, rho_LTM(zvec),c=cosmo_color('LTM'),lw=2.2)

	if labels:
		plt.text(0.335, 0.16, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=70)
		plt.text(0.685, 0.50,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
				transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=72)
		plt.text(0.565, 0.85, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HE'),
			rotation=55)
		plt.text(0.475, 0.35,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=73)
		plt.text(0.43,0.5,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=70)

def plot_distribution_Pk(data_dir='../data/distribution_data/', labels=True):
	xmin, xmax, ymin, ymax = 1e-4, 1, -0.03, 0.03
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.xlabel(r"$k \,\mathrm{[Mpc}^{-1}\mathrm{]}$",fontsize=18)
	plt.ylabel(r"$\left(P(k) - P(k)|_{\Lambda {\rm CDM}}\right)/P(k)|_{\Lambda {\rm CDM}}$",fontsize=18)
	plt.xscale("log")
	plt.yscale("linear")

	file_LCDM_massless  = data_dir + "LCDM_massless_z1_pk.dat"
	file_LCDM  = data_dir + "LCDM_004_z1_pk.dat"
	file_LEDR  = data_dir + "Lownus_004_z1_pk.dat"
	file_HE  = data_dir + "High_nus_004_z1_pk.dat"
	file_HEDR  = data_dir + "High_nus_Mixed_004_z1_pk.dat"
	file_LTM  = data_dir + "LowT_High_nus_004_z1_pk.dat"

	Pk1 = interp1d(np.loadtxt(file_LCDM)[:,0],np.loadtxt(file_LCDM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk2 = interp1d(np.loadtxt(file_LEDR)[:,0],np.loadtxt(file_LEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk3 = interp1d(np.loadtxt(file_HE)[:,0],np.loadtxt(file_HE)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk4 = interp1d(np.loadtxt(file_HEDR)[:,0],np.loadtxt(file_HEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk5 = interp1d(np.loadtxt(file_LTM)[:,0],np.loadtxt(file_LTM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')


	kvec = np.loadtxt(file_LCDM)[:,0]

	plt.plot(0.67 * kvec, (Pk1(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LCDM'),lw=2.2, zorder=10)
	plt.plot(0.67 * kvec, (Pk2(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HE'),lw=2.2)
	plt.plot(0.67 * kvec, (Pk3(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LEDR'),lw=2.2)
	plt.plot(0.67 * kvec, (Pk4(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HEDR'),lw=2.2)
	plt.plot(0.67 * kvec, (Pk5(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LTM'),lw=2.2)

	file_LCDM_massless  = data_dir + "LCDM_massless_z2_pk.dat"
	file_LCDM  = data_dir + "LCDM_004_z2_pk.dat"
	file_LEDR  = data_dir + "Lownus_004_z2_pk.dat"
	file_HE  = data_dir + "High_nus_004_z2_pk.dat"
	file_HEDR  = data_dir + "High_nus_Mixed_004_z2_pk.dat"
	file_LTM  = data_dir + "LowT_High_nus_004_z2_pk.dat"

	Pk1 = interp1d(np.loadtxt(file_LCDM)[:,0],np.loadtxt(file_LCDM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk2 = interp1d(np.loadtxt(file_LEDR)[:,0],np.loadtxt(file_LEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk3 = interp1d(np.loadtxt(file_HE)[:,0],np.loadtxt(file_HE)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk4 = interp1d(np.loadtxt(file_HEDR)[:,0],np.loadtxt(file_HEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk5 = interp1d(np.loadtxt(file_LTM)[:,0],np.loadtxt(file_LTM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')


	kvec = np.loadtxt(file_LCDM)[:,0]

	plt.plot(0.67 * kvec, (Pk2(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HE'),lw=2.2, ls=(1, (5, 1)))
	plt.plot(0.67 * kvec, (Pk3(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LEDR'),lw=2.2, ls=(1, (5, 1)))
	plt.plot(0.67 * kvec, (Pk4(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HEDR'),lw=2.2, ls=(1, (5, 1)))
	plt.plot(0.67 * kvec, (Pk5(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LTM'),lw=2.2, ls=(1, (5, 1)))

	file_LCDM_massless  = data_dir + "LCDM_massless_z3_pk.dat"
	file_LCDM  = data_dir + "LCDM_004_z3_pk.dat"
	file_LEDR  = data_dir + "Lownus_004_z3_pk.dat"
	file_HE  = data_dir + "High_nus_004_z3_pk.dat"
	file_HEDR  = data_dir + "High_nus_Mixed_004_z3_pk.dat"
	file_LTM  = data_dir + "LowT_High_nus_004_z3_pk.dat"

	Pk1 = interp1d(np.loadtxt(file_LCDM)[:,0],np.loadtxt(file_LCDM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk2 = interp1d(np.loadtxt(file_LEDR)[:,0],np.loadtxt(file_LEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk3 = interp1d(np.loadtxt(file_HE)[:,0],np.loadtxt(file_HE)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk4 = interp1d(np.loadtxt(file_HEDR)[:,0],np.loadtxt(file_HEDR)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk5 = interp1d(np.loadtxt(file_LTM)[:,0],np.loadtxt(file_LTM)[:,1],bounds_error=False,fill_value=0.0,kind='linear')


	kvec = np.loadtxt(file_LCDM)[:,0]

	plt.plot(0.67 * kvec, (Pk2(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HE'),lw=2.2, ls=(0, (1, 1)))
	plt.plot(0.67 * kvec, (Pk3(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LEDR'),lw=2.2, ls=(0, (1, 1)))
	plt.plot(0.67 * kvec, (Pk4(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('HEDR'),lw=2.2, ls=(0, (1, 1)))
	plt.plot(0.67 * kvec, (Pk5(kvec) - Pk1(kvec))/Pk1(kvec),cosmo_color('LTM'),lw=2.2, ls=(0, (1, 1)))


	plt.plot([1.5e-4,4e-4], [0.026,0.026],'k',lw=2.2)
	plt.plot([1.5e-4,4e-4], [0.023,0.023],'k',lw=2.2, ls=(1, (5, 1)))
	plt.plot([1.5e-4,4e-4], [0.02,0.02],'k',lw=2.2, ls=(0, (1, 1)))


	plt.text(5e-4,0.0255,r"$z = 0$",fontsize=0.8*16,color='k',rotation=0)
	plt.text(5e-4,0.0225,r"$z = 1.2$",fontsize=0.8*16,color='k',rotation=0)
	plt.text(5e-4,0.0195,r"$z = 3.5$",fontsize=0.8*16,color='k',rotation=0)

	if labels:
		plt.text(0.85, 0.51, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=0)
		plt.text(0.55, 0.37, r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=-35)
		plt.text(0.9, 0.3, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HE'),
			rotation=0)
		plt.text(0.87, 0.405,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=4)
		plt.text(0.87,0.65,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=23)

	plt.text(0.04,0.03,r"$\Omega_\nu / \Omega_\mathrm{m} = 0.009$",transform =plt.gca().transAxes,fontsize=16,color='k',rotation=0)

def plot_distribution_Pk_ratio(data_dir='../data/distribution_data/', labels=True):
	xmin, xmax, ymin, ymax = 1e-4, 1, 0.85, 1.05
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.xlabel(r"$k \,[{\rm h/Mpc}]$",fontsize=1.2*16)
	plt.ylabel(r"$P(k)/P(k)|_{\Lambda {\rm CDM}}^{m_\nu = 0}$",fontsize=1.2*16)
	plt.xscale("log")
	plt.yscale("linear")

	file_0  = data_dir + "LCDM_massless_z1_pk.dat"
	file_1  = data_dir + "LCDM_004_z1_pk.dat"
	file_2  = data_dir + "Lownus_004_z1_pk.dat"
	file_3  = data_dir + "High_nus_004_z1_pk.dat"
	file_4  = data_dir + "High_nus_Mixed_004_z1_pk.dat"
	file_5  = data_dir + "LowT_High_nus_004_z1_pk.dat"
	file_6  = data_dir + "LCDM_003_z1_pk.dat"
	file_7  = data_dir + "LCDM_005_z1_pk.dat"

	Pk0      =interp1d(np.loadtxt(file_0)[:,0],np.loadtxt(file_0)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk1      =interp1d(np.loadtxt(file_1)[:,0],np.loadtxt(file_1)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk2      =interp1d(np.loadtxt(file_2)[:,0],np.loadtxt(file_2)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk3      =interp1d(np.loadtxt(file_3)[:,0],np.loadtxt(file_3)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk4      =interp1d(np.loadtxt(file_4)[:,0],np.loadtxt(file_4)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk5      =interp1d(np.loadtxt(file_5)[:,0],np.loadtxt(file_5)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk6      =interp1d(np.loadtxt(file_6)[:,0],np.loadtxt(file_6)[:,1],bounds_error=False,fill_value=0.0,kind='linear')
	Pk7      =interp1d(np.loadtxt(file_7)[:,0],np.loadtxt(file_7)[:,1],bounds_error=False,fill_value=0.0,kind='linear')


	kvec = np.loadtxt(file_1)[:,0]

	plt.plot(kvec, Pk1(kvec)/Pk0(kvec),cosmo_color('LCDM'),lw=2.2)
	plt.plot(kvec, Pk2(kvec)/Pk0(kvec),cosmo_color('HE'),lw=2.2)
	plt.plot(kvec, Pk3(kvec)/Pk0(kvec),cosmo_color('LEDR'),lw=2.2)
	plt.plot(kvec, Pk4(kvec)/Pk0(kvec),cosmo_color('HEDR'),lw=2.2)
	plt.plot(kvec, Pk5(kvec)/Pk0(kvec),cosmo_color('LTM'),lw=2.2)
	
	plt.plot(kvec, Pk6(kvec)/Pk0(kvec),'k',ls=(1, (5, 1)), lw=0.6)
	plt.plot(kvec, Pk7(kvec)/Pk0(kvec),'k',ls=(1, (5, 1)), lw=0.6)

	plt.fill_between(kvec,Pk7(kvec)/Pk0(kvec),Pk6(kvec)/Pk0(kvec), facecolor='k', linewidth=0, color='k',alpha=0.1,zorder = -5)


	plt.text(0.62,0.52,r"$\Lambda$CDM $\sum m_\nu = (0.12\pm 0.04)\,\mathrm{eV}$",transform =plt.gca().transAxes,fontsize=10,color='k',rotation=-10)

	if labels:
		plt.text(0.88, 0.39, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=-6)
		plt.text(0.60, 0.4, r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=-35)
		plt.text(0.55, 0.64, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HE'),
			rotation=-38)
		plt.text(0.88, 0.325,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=-8)
		plt.text(0.87,0.45,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=-6)

	plt.text(0.02,0.03,r"$\Omega_\nu/\Omega_\mathrm{m}  = 0.009$",transform =plt.gca().transAxes,fontsize=16,color='k',rotation=0)


def plot_distribution_Cl(data_dir='../data/distribution_data/', labels=True, lensed=True):
	if lensed:
		file_LCDM  = data_dir + "LCDM_004_cl_lensed.dat"
		file_LEDR  = data_dir + "Lownus_004_cl_lensed.dat"
		file_HE  = data_dir + "High_nus_004_cl_lensed.dat"
		file_HEDR  = data_dir + "High_nus_Mixed_004_cl_lensed.dat"
		file_LTM  = data_dir + "LowT_High_nus_004_cl_lensed.dat"
	else:
		file_LCDM  = data_dir + "LCDM_004_cl.dat"
		file_LEDR  = data_dir + "Lownus_004_cl.dat"
		file_HE  = data_dir + "High_nus_004_cl.dat"
		file_HEDR  = data_dir + "High_nus_Mixed_004_cl.dat"
		file_LTM  = data_dir + "LowT_High_nus_004_cl.dat"

	plt.plot(np.loadtxt(file_LCDM)[:,0], (np.loadtxt(file_LEDR)[:,1]-np.loadtxt(file_LCDM)[:,1])/np.loadtxt(file_LCDM)[:,1],'r-',
		c=cosmo_color('LEDR'),lw=2.2)
	plt.plot(np.loadtxt(file_LCDM)[:,0], (np.loadtxt(file_HE)[:,1]-np.loadtxt(file_LCDM)[:,1])/np.loadtxt(file_LCDM)[:,1],
		c=cosmo_color('HE'),lw=2.2)
	plt.plot(np.loadtxt(file_LCDM)[:,0], (np.loadtxt(file_HEDR)[:,1]-np.loadtxt(file_LCDM)[:,1])/np.loadtxt(file_LCDM)[:,1],
		c=cosmo_color('HEDR'),lw=2.2)
	plt.plot(np.loadtxt(file_LCDM)[:,0], (np.loadtxt(file_LTM)[:,1]-np.loadtxt(file_LCDM)[:,1])/np.loadtxt(file_LCDM)[:,1],c=cosmo_color('LTM'),lw=2.2)


	plt.plot([2, 2500], [0,0], c=cosmo_color('LCDM'), lw=2.2, zorder=-5)

	if labels:
		plt.text(0.2, 0.47, r"\boldmath{$\Lambda$}\textbf{CDM}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LCDM'),
			rotation=0)
		plt.text(0.55, 0.30,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
				transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LEDR'),
			rotation=70)
		plt.text(0.47, 0.52, r"\textbf{H}\boldmath{$\nu$}",
			transform=plt.gca().transAxes,
			fontsize=12,
			color=cosmo_color('HE'),
			rotation=-15)
		plt.text(0.04, 0.42,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('HEDR'),
			rotation=0)
		plt.text(0.035,0.79,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
			fontsize=10,
			color=cosmo_color('LTM'),
			rotation=0)

def plot_rho_hist(case, nratio=1., data_dir='../data/', step=False):
	folder = data_dir + 'chains_{}/'.format(case)
	print('[utils.py] Loading MCMC samplesn from {}'.format(folder))
	samples = []
	for filepath in glob.iglob(folder + '*__*.txt'):
		try:
			data = np.loadtxt(filepath)
			samples.append(data)
		except OSError:
			print('[utils.py] (WARNING) OSError in {}'.format(folder))
	samples = np.vstack(np.array(samples))
	samples[:,8] *= 3.
	weights = samples[:, 0]
	rho_nonrel = 114 * nratio * samples[:,8]
	hist, bin_edges = np.histogram(rho_nonrel, bins=20, weights=weights)
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	plt.plot(bin_centres, hist/max(hist), 
		color=cosmo_color(case), 
		lw=2.2)
	if step:
		plt.step(bin_centres, hist/max(hist),
			where='mid', 
			color=cosmo_color(case))


def add_cosmo_cases():
	xoff, yoff, offset = 0.15, -0.0, 0.03
	plt.text(0.2 - xoff, 0.90 - yoff,r"\boldmath{$\Lambda$}\textbf{CDM}",transform=plt.gca().transAxes,
		fontsize=14,
		color=cosmo_color('LCDM'),
		rotation=0)
	plt.text(0.35, 0.09, r"\boldmath{$\Lambda$}\textbf{CDM}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LCDM'),
		rotation=42)
	plt.text(0.38 - xoff, 0.89 - yoff,r"$\sum m_\nu = 0.12\,\mathrm{eV}$, $N_{\rm eff}^\nu = 3.044$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)

	plt.text(0.2 - xoff, 0.82 + offset - yoff,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
		fontsize=14,
		color=cosmo_color('LEDR'),
		rotation=0)
	plt.text(0.11, 0.09,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
			transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LEDR'),
		rotation=62)
	plt.text(0.38 - xoff, 0.81 + offset - yoff,r"$\sum m_\nu = 0.12\,\mathrm{eV}$, $N_{\rm eff}^\nu = 0.5$, $N_{\rm eff}^{\rm DR} = 2.544$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)

	plt.text(0.2 - xoff, 0.74 + 2 * offset - yoff, r"\textbf{H}\boldmath{$\nu$}",
		transform=plt.gca().transAxes,
		fontsize=14,
		color=cosmo_color('HE'),
		rotation=0)
	plt.text(0.77, 0.32, r"\textbf{H}\boldmath{$\nu$}",
		transform=plt.gca().transAxes,
		fontsize=12,
		color=cosmo_color('HE'),
		rotation=80)
	plt.text(0.38 - xoff, 0.73 + 2 * offset - yoff,r"$\sum m_\nu = 1.20\,\mathrm{eV}$, $N_{\rm eff}^\nu = 3.044$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)

	plt.text(0.2 - xoff, 0.66 + 3 * offset - yoff,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=14,
		color=cosmo_color('HEDR'),
		rotation=0)
	plt.text(0.61, 0.09,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('HEDR'),
		rotation=63)
	plt.text(0.38 - xoff, 0.65 + 3 * offset - yoff,r"$\sum m_\nu = 1.20\,\mathrm{eV}$, $N_{\rm eff}^\nu = 1.5$, $N_{\rm eff}^{\rm DR} = 1.544$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)

	plt.text(0.2 - xoff, 0.58 + 4 * offset - yoff,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
		fontsize=14,
		color=cosmo_color('LTM'),
		rotation=0)
	plt.text(0.455,0.24,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LTM'),
		rotation=83)
	plt.text(0.38 - xoff, 0.57 + 4 * offset - yoff,r"$\sum  m_\nu = 0.12\,\mathrm{eV}$, $N_{\rm eff}^{\nu} = 3.044$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)

	plt.text(0.2 - xoff, 0.55 + 2 * offset - yoff,r"$\Omega_\nu / \Omega_{\rm m} = 0.009$, $N_{\rm eff} = 3.044$",
		transform=plt.gca().transAxes,
		fontsize=16,
		color='k',
		rotation=0)

def plot_CMB_sensitivity(case, expt, data_dir='../data/', ls='-', alpha=1.0):
	planck = np.loadtxt(data_dir + 'planck_sensitivity.txt')
	cmbs4 = np.loadtxt(data_dir + 'cmb-s4_sensitivity.txt')
	case_dict = {'LCDM': 1,
				 'HEDR': 2,
				 'LTM': 3,
				 'HE': 4,
				 'LEDR': 5}
	if expt.lower()[0] == 'p':
		plt.plot(3 * 114 * planck[:, 0], planck[:, case_dict[case]],
			c=cosmo_color(case),
			lw=2.2, ls=ls, alpha=alpha)
	elif expt.lower()[0] == 'c':
		plt.plot(3 * 114 * cmbs4[:, 0], cmbs4[:, case_dict[case]],
			c=cosmo_color(case),
			lw=2.2,
			ls='-')

def plot_main_contour(MCMC_run='DistNu', data_dir='../data/'):
	folder = data_dir + 'chains_{}/'.format(MCMC_run)
	print('[utils.py] Loading MCMC data from {}'.format(folder))
	idc = {
		'omegab': 2,
		'omegacdm': 3,
		'100thetas': 4,
		'ln10^10As': 5,
		'ns': 6,
		'mncdm': 8,
		'Neffncdm': 10,
		'logystar': 11,
		'logsigmastar': 12,
		'Nur': 9,
		'OmegaLambda': 35,
		'H0': 36,
		'sigma8': 38
	}
	samples = []
	for filepath in glob.iglob(folder + '*__*.txt'):
		try:
			data = np.loadtxt(filepath)
			if len(data) != 0:
				samples.append(data)
		except OSError:
			print('[utils.py] (WARNING) OSError in {}'.format(folder))
	samples = np.vstack(np.array(samples))
	weights = samples[:, 0]
	Neff = samples[:, idc['Neffncdm']] + samples[:, idc['Nur']]
	rho_nonrel = 114. * n_ratio(samples[:, idc['Neffncdm']], 10**samples[:, idc['logystar']], 10**samples[:, idc['logsigmastar']]) * samples[:, idc['mncdm']] * 3.
	summnu = 3. * samples[:, idc['mncdm']]
	
	ax = plt.subplot(3, 3, 1)
	hist, bin_edges = np.histogram(summnu, bins=20, weights=weights)
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	plt.plot(bin_centres, hist/max(hist), 
		color=cosmo_color('LCDM'), 
		lw=2.2)
	plt.step(bin_centres, hist/max(hist), 
		where='mid', 
		color=cosmo_color('LCDM'))
	
	plt.xlim(0, 4)
	ax.set_yticklabels([])
	plt.xticks([0, 1, 2, 3, 4], fontsize=22)
	ax.set_title(r'$\sum m_\nu \, \mathrm{[eV]}$', pad=15, fontsize=24)
	ax.xaxis.set_label_position('top')
	ax.tick_params(labelbottom=False,labeltop=True)
	

	ax = plt.subplot(3, 3, 5)
	hist, bin_edges = np.histogram(Neff, bins=20, weights=weights)
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	plt.plot(bin_centres, hist/max(hist), 
		color=cosmo_color('LCDM'), 
		lw=2.2)
	plt.step(bin_centres, hist/max(hist), 
		where='mid', 
		color=cosmo_color('LCDM'))
	
	plt.xlim(2, 4)
	ax.set_yticklabels([])
	ax.set_title(r'$N_\mathrm{eff}$', pad=15, fontsize=24)
	ax.xaxis.set_label_position('top')
	ax.tick_params(labelbottom=False,labeltop=True)
	plt.axvline(2.6, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(2.79, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(2.99, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	plt.axvline(3.18, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(3.43, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.xticks([2, 2.5, 3, 3.5, 4], fontsize=22)

	ax = plt.subplot(3, 3, 9)
	hist, bin_edges = np.histogram(rho_nonrel, bins=200, weights=weights)
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	plt.plot(bin_centres, hist/max(hist), 
		color=cosmo_color('LCDM'), 
		lw=2.2)
	plt.step(bin_centres, hist/max(hist), 
		where='mid', 
		color=cosmo_color('LCDM'))
	plt.xlim(0, 40)
	ax.set_yticklabels([])
	ax.set_xlabel(r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$', fontsize=24)
	ax.set_title(r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$', pad=15, fontsize=24)
	ax.xaxis.set_label_position('bottom')
	ax.tick_params(labelbottom=True,labeltop=True)
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	plt.xticks([0, 10, 20, 30, 40], fontsize=22)

	ax = plt.subplot(3, 3, 4)
	plot_hist2d(summnu, Neff, interpolation_smoothing=40., gaussian_smoothing=1.8)
	plt.xlim(0, 4)
	plt.xticks([0, 1, 2, 3, 4], fontsize=22)
	plt.ylim(2, 4)
	ax.set_xticklabels([])
	ax.set_ylabel(r'$N_\mathrm{eff}$', fontsize=24)
	plt.axhline(2.6, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.axhline(2.79, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axhline(2.99, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	plt.axhline(3.18, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axhline(3.43, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.yticks([2, 2.5, 3, 3.5, 4], fontsize=22)


	ax = plt.subplot(3, 3, 7)
	plot_hist2d(summnu, rho_nonrel)
	plt.xlim(0, 4)
	plt.xticks([0, 1, 2, 3, 4], fontsize=22)
	plt.ylim(0, 40)
	plt.yticks([0, 10, 20, 30, 40], fontsize=22)
	ax.set_xlabel(r'$\sum m_\nu \, \mathrm{[eV]}$', fontsize=24)
	ax.set_ylabel(r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$', fontsize=24)
	plt.axhline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)


	ax = plt.subplot(3, 3, 8)
	plot_hist2d(Neff, rho_nonrel)
	plt.xlim(2, 4)
	plt.xticks([2, 2.5, 3, 3.5, 4], fontsize=22)
	plt.ylim(0, 40)
	plt.yticks([0, 10, 20, 30, 40], fontsize=22)
	ax.set_yticklabels([])
	ax.set_xlabel(r'$N_\mathrm{eff}$', fontsize=24)
	plt.axvline(2.6, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(2.79, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(2.99, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	plt.axvline(3.18, c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
	plt.axvline(3.43, c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	plt.axhline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)

	plt.subplots_adjust(hspace=0.1, wspace=0.1)

def plot_1d_posterior(array, weights, name, subplot=[1, 1, 1], nbins=20, xlim=None, xticks=True, yticks=True, bottomlabel=False, toplabel=True, xtickpts=None, ytickpts=None):
	ax = plt.subplot(subplot[0], subplot[1], subplot[2])
	hist, bin_edges = np.histogram(array, bins=nbins, weights=weights)
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	plt.plot(bin_centres, hist/max(hist), 
		color=cosmo_color('LCDM'), 
		lw=2.2)
	plt.step(bin_centres, hist/max(hist), 
		where='mid', 
		color=cosmo_color('LCDM'))
	if type(xlim) != type(None):
		plt.xlim(xlim[0], xlim[1])
	if not yticks:
		ax.set_yticklabels([])
	if not xticks:
		ax.set_xticklabels([])
	ax.set_title(name, pad=15, fontsize=18)
	ax.xaxis.set_label_position('top')
	ax.tick_params(labelbottom=bottomlabel,labeltop=toplabel)
	if bottomlabel:
		ax.set_xlabel(name)
		ax.xaxis.set_label_position('bottom')
	if type(xtickpts) != type(None):
		ax.set_xticks(xtickpts)
	if type(ytickpts) != type(None):
		ax.set_yticks(ytickpts)


def plot_appendix_contour(MCMC_run='DistNu', data_dir='../data/'):
	folder = data_dir + 'chains_{}/'.format(MCMC_run)
	print('[utils.py] Loading MCMC data from {}'.format(folder))
	idc = {
		'omegab': 2,
		'omegacdm': 3,
		'100thetas': 4,
		'ln10^10As': 5,
		'ns': 6,
		'mncdm': 8,
		'Neffncdm': 10,
		'logystar': 11,
		'logsigmastar': 12,
		'Nur': 9,
		'OmegaLambda': 35,
		'H0': 36,
		'sigma8': 38
	}
	samples = []
	for filepath in glob.iglob(folder + '*__*.txt'):
		try:
			data = np.loadtxt(filepath)
			if len(data) != 0:
				samples.append(data)
		except OSError:
			print('[utils.py] (WARNING) OSError in {}'.format(folder))
	samples = np.vstack(np.array(samples))
	weights = samples[:, 0]
	Neff = samples[:, idc['Neffncdm']] + samples[:, idc['Nur']]
	rho_nonrel = 114. * n_ratio(samples[:, idc['Neffncdm']], 10**samples[:, idc['logystar']], 10**samples[:, idc['logsigmastar']]) * samples[:, idc['mncdm']] * 3.
	summnu = 3. * samples[:, idc['mncdm']]

	# get_bounds(values=samples[:, idc['H0']], weights=weights)
	# get_bounds(values=samples[:, idc['sigma8']], weights=weights)
	Neff_bounds = [2.6, 2.79, 2.99, 3.18, 3.43]
	H0_bounds = [65.43, 66.54, 67.72, 68.85, 70.20]
	sigma8_bounds = [0.79, 0.80, 0.81, 0.82, 0.84]


	plot_1d_posterior(array=Neff, weights=weights, name=r'$N_\mathrm{eff}$', subplot=[6, 6, 1], nbins=20, xlim=(2, 4), xticks=True, yticks=False, bottomlabel=False, toplabel=True)
	add_sigma_lines(bounds=Neff_bounds, axis='x')
	
	plot_1d_posterior(array=rho_nonrel, weights=weights, name=r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$', subplot=[6, 6, 8], nbins=200, xlim=(0, 40), xticks=True, yticks=False, bottomlabel=False, toplabel=True)
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	plt.xticks([0, 10, 20, 30, 40])
	
	plot_1d_posterior(array=samples[:, idc['logystar']], weights=weights, name=r'$\log_{10}y_\star$', subplot=[6, 6, 15], nbins=20, xlim=(-1, 2), xticks=True, yticks=False, bottomlabel=False, toplabel=True, xtickpts=[-1, 0, 1, 2])
	
	plot_1d_posterior(array=samples[:, idc['logsigmastar']], weights=weights, name=r'$\log_{10}\sigma_\star$', subplot=[6, 6, 22], nbins=20, xlim=(-1, 2), xticks=True, yticks=False, bottomlabel=False, toplabel=True,xtickpts=[-1, 0, 1, 2])
	
	plot_1d_posterior(array=samples[:, idc['H0']], weights=weights, name=r'$H_0$', subplot=[6, 6, 29], nbins=150, xlim=(60, 80), xticks=True, yticks=False, bottomlabel=False, toplabel=True)
	add_sigma_lines(bounds=H0_bounds, axis='x')

	plot_1d_posterior(array=samples[:, idc['sigma8']], weights=weights, name=r'$\sigma_8$', subplot=[6, 6, 36], nbins=150, xlim=(0.7, 0.9), xticks=True, yticks=False, bottomlabel=True, toplabel=True)
	add_sigma_lines(bounds=sigma8_bounds, axis='x')

	ax = plt.subplot(6, 6, 7)
	plot_hist2d(Neff, rho_nonrel)
	plt.xlim(2, 4)
	plt.ylim(0, 40)
	plt.yticks([0, 10, 20, 30, 40])
	ax.set_xticklabels([])
	ax.set_ylabel(r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$')
	add_sigma_lines(bounds=Neff_bounds, axis='x')
	plt.axhline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)

	ax = plt.subplot(6, 6, 13)
	plot_hist2d(Neff, samples[:, idc['logystar']])
	plt.xlim(2, 4)
	plt.ylim(-1, 2)
	ax.set_xticklabels([])
	ax.set_yticks([-1, 0, 1, 2])
	ax.set_ylabel(r'$\log_{10}y_\star$')
	add_sigma_lines(bounds=Neff_bounds, axis='x')

	ax = plt.subplot(6, 6, 19)
	plot_hist2d(Neff, samples[:, idc['logsigmastar']])
	plt.xlim(2, 4)
	plt.ylim(-1, 2)
	ax.set_xticklabels([])
	ax.set_yticks([-1, 0, 1, 2])
	ax.set_ylabel(r'$\log_{10}\sigma_\star$')
	add_sigma_lines(bounds=Neff_bounds, axis='x')

	ax = plt.subplot(6, 6, 25)
	plot_hist2d(Neff, samples[:, idc['H0']])
	plt.xlim(2, 4)
	plt.ylim(60, 80)
	ax.set_xticklabels([])
	ax.set_ylabel(r'$H_0$')
	add_sigma_lines(bounds=Neff_bounds, axis='x')
	add_sigma_lines(bounds=H0_bounds, axis='y')

	ax = plt.subplot(6, 6, 31)
	plot_hist2d(Neff, samples[:, idc['sigma8']])
	plt.xlim(2, 4)
	plt.ylim(0.7, 0.9)
	ax.set_xlabel(r'$N_\mathrm{eff}$')
	ax.set_ylabel(r'$\sigma_8$')
	add_sigma_lines(bounds=Neff_bounds, axis='x')
	add_sigma_lines(bounds=sigma8_bounds, axis='y')

	ax = plt.subplot(6, 6, 14)
	plot_hist2d(rho_nonrel, samples[:, idc['logystar']])
	plt.xlim(0, 40)
	plt.xticks([0, 10, 20, 30, 40])
	plt.ylim(-1, 2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	
	ax = plt.subplot(6, 6, 20)
	plot_hist2d(rho_nonrel, samples[:, idc['logsigmastar']])
	plt.xlim(0, 40)
	plt.xticks([0, 10, 20, 30, 40])
	plt.ylim(-1, 2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)

	ax = plt.subplot(6, 6, 26)
	plot_hist2d(rho_nonrel, samples[:, idc['H0']])
	plt.xlim(0, 40)
	plt.xticks([0, 10, 20, 30, 40])
	plt.ylim(60, 80)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	add_sigma_lines(bounds=H0_bounds, axis='y')

	ax = plt.subplot(6, 6, 32)
	plot_hist2d(rho_nonrel, samples[:, idc['sigma8']])
	plt.xlim(0, 40)
	plt.xticks([0, 10, 20, 30, 40])
	plt.ylim(0.7, 0.9)
	ax.set_yticklabels([])
	ax.set_xlabel(r'$\rho_{\nu, 0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$')
	plt.axvline(14.0, c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
	add_sigma_lines(bounds=sigma8_bounds, axis='y')

	ax = plt.subplot(6, 6, 21)
	plot_hist2d(samples[:, idc['logystar']], samples[:, idc['logsigmastar']], num_bins=30)
	plt.xlim(-1, 2)
	plt.ylim(-1, 2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	ax = plt.subplot(6, 6, 27)
	plot_hist2d(samples[:, idc['logystar']], samples[:, idc['H0']])
	plt.xlim(-1, 2)
	plt.ylim(60, 80)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	add_sigma_lines(bounds=H0_bounds, axis='y')

	ax = plt.subplot(6, 6, 33)
	plot_hist2d(samples[:, idc['logystar']], samples[:, idc['sigma8']])
	plt.xlim(-1, 2)
	plt.ylim(0.7, 0.9)
	ax.set_yticklabels([])
	ax.set_xticks([-1, 0, 1, 2])
	ax.set_xlabel(r'$\log_{10}y_\star$')
	add_sigma_lines(bounds=sigma8_bounds, axis='y')

	ax = plt.subplot(6, 6, 28)
	plot_hist2d(samples[:, idc['logsigmastar']], samples[:, idc['H0']])
	plt.xlim(-1, 2)
	plt.ylim(60, 80)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	add_sigma_lines(bounds=H0_bounds, axis='y')

	ax = plt.subplot(6, 6, 34)
	plot_hist2d(samples[:, idc['logsigmastar']], samples[:, idc['sigma8']])
	plt.xlim(-1, 2)
	plt.ylim(0.7, 0.9)
	ax.set_yticklabels([])
	ax.set_xticks([-1, 0, 1, 2])
	ax.set_xlabel(r'$\log_{10}\sigma_\star$')
	add_sigma_lines(bounds=sigma8_bounds, axis='y')

	ax = plt.subplot(6, 6, 35)
	plot_hist2d(samples[:, idc['H0']], samples[:, idc['sigma8']])
	plt.xlim(60, 80)
	plt.ylim(0.7, 0.9)
	ax.set_yticklabels([])
	ax.set_xlabel(r'$H_0$')
	add_sigma_lines(bounds=H0_bounds, axis='x')
	add_sigma_lines(bounds=sigma8_bounds, axis='y')

	plt.subplots_adjust(hspace=0.18, wspace=0.18)

def add_sigma_lines(bounds, axis):
	if axis == 'y':
		plt.axhline(bounds[0], c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
		plt.axhline(bounds[1], c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
		plt.axhline(bounds[2], c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
		plt.axhline(bounds[3], c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
		plt.axhline(bounds[4], c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
	elif axis == 'x':
		plt.axvline(bounds[0], c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)
		plt.axvline(bounds[1], c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
		plt.axvline(bounds[2], c=cosmo_color('HEDR'), lw=2.8, zorder=-10)
		plt.axvline(bounds[3], c='k', lw=1.2, ls=(1, (5, 1)), zorder=-10)
		plt.axvline(bounds[4], c='k', lw=0.8, ls=(1, (5, 1)), zorder=-10)


def get_bounds(values, weights, symmetric=True):
	values_sort = np.vstack([weights, values]).T
	values_sort = values_sort[values_sort[:, 1].argsort()]
	bounds = []
	if symmetric:
		for level in [0.025, 0.16, 0.5, 0.82, 0.975]:
			mask = (np.cumsum(values_sort[:, 0])/np.sum(weights) < level)
			val = values_sort[mask, 1][-1]
			bounds.append(val)
		print('\t-2sig\t|\t-1sig\t|\tMid\t|\t1sig\t|\t2sig')
		print('\t{:.2f}\t|\t{:.2f}\t|\t{:.2f}\t|\t{:.2f}\t|\t{:.2f}'.format(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4]))
	else:
		for level in [0.68, 0.95]:
			mask = (np.cumsum(values_sort[:, 0])/np.sum(weights) < level)
			val = values_sort[mask, 1][-1]
			bounds.append(val)
		print('\t1sig\t|\t2sig')
		print('\t{:.2f}\t|\t{:.2f}'.format(bounds[0], bounds[1]))


def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return hist, bin_edges, bin_centres

def get_hist2d(datax, datay, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(datax, datay, bins=num_bins, weights=weights)
    bin_centresx = 0.5*(bin_edgesx[1:]+bin_edgesx[:-1])
    bin_centresy = 0.5*(bin_edgesy[1:]+bin_edgesy[:-1])
    return hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy

def get_hist2d_contour(datax, datay, num_bins=40, weights=[None]):
	if not any(weights):
		weights = np.ones(len(datax))
	hist, bin_edgesx, bin_edgesy = np.histogram2d(datax, datay, bins=num_bins, weights=weights)
	bin_centresx = 0.5*(bin_edgesx[1:]+bin_edgesx[:-1])
	bin_centresy = 0.5*(bin_edgesy[1:]+bin_edgesy[:-1])
	hist2 = hist.min() + np.zeros((hist.shape[0] + 4, hist.shape[1] + 4))
	hist2[2:-2, 2:-2] = hist
	hist2[2:-2, 1] = hist[:, 0]
	hist2[2:-2, -2] = hist[:, -1]
	hist2[1, 2:-2] = hist[0]
	hist2[-2, 2:-2] = hist[-1]
	hist2[1, 1] = hist[0, 0]
	hist2[1, -2] = hist[0, -1]
	hist2[-2, 1] = hist[-1, 0]
	hist2[-2, -2] = hist[-1, -1]
	bin_centresx2 = np.concatenate(
	    [
	        bin_centresx[0] + np.array([-2, -1]) * np.diff(bin_centresx[:2]),
	        bin_centresx,
	        bin_centresx[-1] + np.array([1, 2]) * np.diff(bin_centresx[-2:]),
	    ]
	)
	bin_centresy2 = np.concatenate(
	    [
	        bin_centresy[0] + np.array([-2, -1]) * np.diff(bin_centresy[:2]),
	        bin_centresy,
	        bin_centresy[-1] + np.array([1, 2]) * np.diff(bin_centresy[-2:]),
	    ]
	)
	return hist2, bin_edgesx, bin_edgesy, bin_centresx2, bin_centresy2

def ctr_level(hist, lvl, infinite=False):
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist = cum_hist / cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)
    clist = [0]+[hist[-i] for i in alvl]
    if not infinite:
        return clist[1:]
    return clist

def ctr_level2d(histogram2d, lvl, infinite=False):
    hist = histogram2d.flatten()*1.
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0]+[hist[-i] for i in alvl]+[hist.max()]
    if not infinite:
        return clist[1:]
    return clist

def plot_hist2d(datax, datay, num_bins=40, weights=[None], color=cosmo_color('LCDM'), interpolation_smoothing=4., gaussian_smoothing=0.8):
    if not any(weights):
        weights = np.ones(len(datax))
    if color == None:
        color="black"

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(datax, datay, num_bins=num_bins, weights=weights)

    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(bin_centresy, interpolation_smoothing, mode='reflect')
    interp_x_centers = scipy.ndimage.zoom(bin_centresx,interpolation_smoothing, mode='reflect')
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode='reflect')
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(interp_hist, [sigma,sigma], mode='reflect')

    density_cmap = LinearSegmentedColormap.from_list("density_cmap", [color, (1, 1, 1, 0)]).reversed()
    plt.imshow(np.transpose(interp_smoothed_hist)[::-1], extent=[bin_edgesx.min(), bin_edgesx.max(), bin_edgesy.min(), bin_edgesy.max()], interpolation="nearest", cmap=density_cmap, aspect="auto")

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d_contour(datax, datay, num_bins=num_bins, weights=weights)

    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(bin_centresy, interpolation_smoothing, mode='reflect')
    interp_x_centers = scipy.ndimage.zoom(bin_centresx,interpolation_smoothing, mode='reflect')
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode='reflect')
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(interp_hist, [sigma,sigma], mode='reflect')

    plt.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=color, linewidths=2.0, levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), linestyles='-')

if __name__ == '__main__':
	print('[utils.py] Starting utils.py')
	plot_main_contour()
