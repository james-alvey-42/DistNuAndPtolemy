import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.special import zeta

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

def n_FD(Tnu=1.95):
	K_TO_CM = 4.366
	return 3 * zeta(3) / (4 * np.pi**2) * Tnu**3 * K_TO_CM**3

def delta_LCDM(mlight_eV):
	return 76.5 * mlight_eV**(2.21)

def load_ptolemy(Tyrs=1.0, Delta=100.0, mT=100.0, Gammab=1e-5, order='normal', spin='Dirac', filename=None, data_dir='../data/'):
	if filename is None:
		filename = '[t]{}_[d]{}_[m]{}_[o]{}_[s]{}_[b]{}.txt'.format(Tyrs, Delta, mT, order.lower()[0], spin.lower()[0], Gammab)
	try:
		data = np.loadtxt(data_dir + filename)
		return data
	except OSError:
		print('[utils.py] (ERROR) Cannot find file:', data_dir + filename)

def plot_ptolemy(data, gauss_filter=1.5, ctr_labels=True):
	mlight, nloc, sensitivity = data[:, 0], data[:, 1], data[:, 2]
	num_m = len(np.unique(mlight))
	num_n = len(np.unique(nloc))
	M, N, S = mlight.reshape(num_n, num_m), nloc.reshape(num_n, num_m), sensitivity.reshape(num_n, num_m)
	
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
		plt.clabel(CS, CS.levels[:2], inline=True, inline_spacing=35, fmt=fmt, fontsize=10, manual=[(717, 15), (717, 33)])
		plt.clabel(CS, CS.levels[2:], inline=True, inline_spacing=45, fmt=fmt, fontsize=10, manual=[(670, 56)])
	plt.text(250.0, 240.0, 'PTOLEMY', fontsize=14, color='lightgreen')
	plt.xscale('log')
	plt.yscale('log')

def add_case_labels(Tyrs=1, Delta=100, mT=100, Gammab=1e-5, order='Normal', spin='Dirac', xmin=12.0):
	plt.text(xmin, 6.2, r'$\mathrm{' + r'{}'.format(order) + r'\ ordering}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 4.7, r'$\mathrm{' + r'{}'.format(spin) + r'}\ \nu$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 3.3, r'$T = ' + r'{}'.format(int(Tyrs)) + r'\, \mathrm{yr}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 2.4, r'$\Delta = ' + r'{}'.format(int(Delta)) + r'\, \mathrm{meV}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.75, r'$M_\mathrm{T} ='  + r'{}'.format(int(mT)) + r'\ \mathrm{g}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.3, r'$\Gamma_\mathrm{b} = 10^{'  + r'{}'.format(int(np.log10(Gammab))) + r'}\ \mathrm{Hz}$', fontsize=12, color='k', rotation=0)

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

def plot_highp(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	
	plt.plot(mlight_arr, n_cosmo_max_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.text(250.0, 4.1, r'$\langle p_\nu\rangle \gg 3T$', fontsize=10, rotation=-32, color="#3F7BB6")

	plt.fill_between(mlight_arr, n_cosmo_max_arr, np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", edgecolor="#3F7BB6", linewidth=0.0, zorder=-10, alpha=0.2)
	plt.fill_between(mlight_arr, n_cosmo_max_arr, np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", hatch="---", edgecolor="#d9e5f0", linewidth=0.0, zorder=-9, alpha=1.)

def plot_lowT(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	Tnu_arr = 1.95 * np.power(n_cosmo_max_arr/n_FD(), 1./3.)
	
	clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering.txt', unpack=True)
	delta_fn = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.text(10.8, 73.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=10, rotation=-18, color="purple")
	
	plt.fill_between(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", edgecolor="purple", linewidth=0.0, zorder=-10, alpha=0.2)
	plt.fill_between(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), np.repeat(1e3, len(mlight_arr)), facecolor="none", color="none", hatch="|||", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=1.)
	
	plt.text(21.0, 325.0, 'CMB', fontsize=12, color='k')
	plt.text(21.0, 240.0, 'Excluded', fontsize=12, color='k')

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
			s=200,
			linewidths=0.1,
			edgecolors='k',
			zorder=9
			)
	plt.text(12.0, 44.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=10, color="darkgoldenrod")

def add_KATRIN(forecast=False):
	if not forecast:
		plt.axvline(200.0, c='#BF4145', zorder=-9, ls=(1, (5, 1)), lw=1.2)
		plt.text(170.0, 1.3, 'KATRIN Sensitivity', fontsize=9, rotation=90, color="#BF4145")
		plt.annotate(s='', xytext=(205.0, 2.8), xy=(350.0, 2.8), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': '#BF4145'}, color='#BF4145')
	else:
		plt.axvline(200.0, c='#BF4145', zorder=-10, ls='-', lw=1.2)
		plt.fill_betweenx([1, 1000], 200.0, 1000.0, facecolor="none", color="none", hatch="xxxx", edgecolor='#f2d9da', linewidth=0.0, zorder=-11, alpha=1.)
		plt.text(174.0, 1.3, 'KATRIN', fontsize=9, rotation=90, color="#BF4145")


def add_0vvb():
	plt.axvline(480.0, c='#d37c2d', zorder=-11, lw=1.2)
	plt.text(405.0, 1.7, r'$0\nu\beta\beta$', fontsize=9, rotation=90, color="#d37c2d")
	plt.fill_betweenx([1, 1000], 480.0, 1000.0, facecolor="none", color="none", hatch="xxxx", edgecolor='#d37c2d', linewidth=0.0, zorder=-11, alpha=0.2)
	plt.annotate(s='', xytext=(490.0, 1.9), xy=(834.0, 1.9), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': '#d37c2d'}, color='#d37c2d')

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
		plt.clabel(CS, CS.levels[:2], inline=True, inline_spacing=35, fmt=fmt, fontsize=10, manual=[(717, 15), (717, 33)])
		plt.clabel(CS, CS.levels[2:], inline=True, inline_spacing=45, fmt=fmt, fontsize=10, manual=[(670, 56)])
	plt.text(250.0, 240.0, 'PTOLEMY', fontsize=14, color='lightgreen')
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
	plt.text(92.0, 3.65, r'$\langle p_\nu\rangle \gg 3T$', fontsize=10, rotation=-28, color="#3F7BB6")

def plot_lowT_DESI(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	sum_mnu_arr = np.geomspace(1e-3 * 10, 1e3 * 1000, 1000)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.04 / sum_mnu_arr)
	Tnu_upper_arr = 1.95 * np.power(n_upper_arr/n_FD(), 1./3.)
	Tnu_lower_arr = 1.95 * np.power(n_lower_arr/n_FD(), 1./3.)
	
	clus_data_mlight, clus_data_delta, _ = np.loadtxt(data_dir + 'lowT_clustering_008.txt', unpack=True)
	delta_fn_upper = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	clus_data_mlight, clus_data_delta, _ = np.loadtxt(data_dir + 'lowT_clustering_004.txt', unpack=True)
	delta_fn_lower = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_upper((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_upper_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_lower_arr), (1 + delta_fn_upper((1./3.) * 1e3 * sum_mnu_arr)) * n_FD(Tnu_upper_arr), facecolor="purple", color="purple", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=0.2)
	plt.text(63, 19.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=10, rotation=-23, color="purple")
	plt.text(31.0, 13.4, 'DESI/EUCLID', fontsize=12, rotation=-32, color='#faf9fc')

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
	plt.text(26., 40.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=10, rotation=-31, color="darkgoldenrod")

def add_mlightest_zero(order='normal'):
	plt.axvline((1./3.) * 1e3 * sum_mnu(mlightest_eV=0.0, order=order), c='k', zorder=-12, ls=(1, (5, 1)), lw=1.1, alpha=0.4)
	plt.text(21, 300, r'$m_\mathrm{lightest} \geq 0$', rotation=-90, fontsize=8)
	plt.annotate(s='', xytext=(24.5, 500.0), xy=(42.0, 500.0), arrowprops={'arrowstyle': '-|>', 'lw':1.0, 'color': 'k'}, color='k', alpha=0.8)

def plot_highp_simple(order='normal'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	
	plt.plot(mlight_arr, n_cosmo_max_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.text(250.0, 4.1, r'$\langle p_\nu\rangle \gg 3T$', fontsize=10, rotation=-32, color="#3F7BB6")

def plot_lowT_simple(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(10, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_cosmo_max_arr = n_FD() * (0.12 / sum_mnu_arr)
	Tnu_arr = 1.95 * np.power(n_cosmo_max_arr/n_FD(), 1./3.)
	
	clus_data_mlight, clus_data_delta = np.loadtxt(data_dir + 'lowT_clustering.txt', unpack=True)
	delta_fn = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot(mlight_arr, (1 + delta_fn(mlight_arr)) * n_FD(Tnu_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.text(400, 80.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=10, rotation=41, color="purple", zorder=9)


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
			zorder=6
			)
	plt.text(120, 130.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=10, color="darkgoldenrod", rotation=40, zorder=9)


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
	Nb_data = 1.05189 * (gamma_b/1e-5)
	Ei_arr = np.linspace(pta.Eend0() - 5000, pta.Eend0() + 10000., int(15000/delta))

	N_data_arr = pta.N_total(Ei_arr, Tyrs, delta, mlight, nloc, NT, order=order, DEEnd=0.0, gamma_b=gamma_b, cDM=cDM_sim)
	N_beta_arr = pta.N_beta(Ei_arr, Tyrs, delta, mlight, NT, order, DEEnd=0.0)
	N_CNB_arr = pta.N_CNB(Ei_arr, Tyrs, delta, mlight, nloc, NT, order, DEEnd=0.0, cDM=cDM_sim)
	N_CNB_large_arr = pta.N_CNB(Ei_arr, Tyrs, delta, mlight, 1e2 * nloc, NT, order, DEEnd=0.0, cDM=cDM_sim)
	N_background_arr = N_data_arr - N_beta_arr - N_CNB_arr

	print('[utils.py] Computed events for mlight = {} meV, delta = {} meV'.format(mlight, delta))
	return {'energies': Ei_arr, 'beta': N_beta_arr, 'CNB': N_CNB_arr, 'CNB_large': N_CNB_large_arr, 'background': N_background_arr}

def plot_all_events(arr_dict):
	Ei_arr = arr_dict['energies']
	N_background_arr = arr_dict['background']
	N_beta_arr = arr_dict['beta']
	N_CNB_arr = arr_dict['CNB']
	N_CNB_large_arr = arr_dict['CNB_large']
	plot_binned_data(Ei_arr, N_background_arr, Ei_arr[-1] - Ei_arr[-2], step_color='k', fill_color='k', zorder=0, points=False)	
	plot_binned_data(Ei_arr, N_beta_arr, Ei_arr[-1] - Ei_arr[-2], step_color='#3F7BB6', fill_color='#3F7BB6', zorder=2)
	plot_binned_data(Ei_arr, N_CNB_large_arr, Ei_arr[-1] - Ei_arr[-2], step_color='darkgoldenrod', fill_color='darkgoldenrod', zorder=4)
	plot_binned_data(Ei_arr, N_CNB_arr, Ei_arr[-1] - Ei_arr[-2], step_color='purple', fill_color='purple', zorder=6)
	set_xy_scales()
	set_xy_lims(-2000, 1000, 1e-2, 1e14)
	plt.xticks([-2000, -1000, 0, 1000])
	plt.yticks([1e-2, 1e2, 1e6, 1e10, 1e14])

def plot_snr(arr_dict):
	Ei_arr = arr_dict['energies']
	N_background_arr = arr_dict['background']
	N_beta_arr = arr_dict['beta']
	N_CNB_arr = arr_dict['CNB']
	N_CNB_large_arr = arr_dict['CNB_large']
	plot_binned_data(Ei_arr, N_CNB_large_arr/np.sqrt(N_beta_arr + N_background_arr), Ei_arr[-1] - Ei_arr[-2], step_color='darkgoldenrod', fill_color='darkgoldenrod')
	plot_binned_data(Ei_arr, N_CNB_arr/np.sqrt(N_beta_arr + N_background_arr), Ei_arr[-1] - Ei_arr[-2], step_color='purple', fill_color='purple')
	plt.text(-1900, 2.0, r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}} > 1$', c='k', fontsize=12, rotation=0)
	plt.annotate(s='', xytext=(-300.0, 1.1), xy=(-300.0, 60.0), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': 'k'}, color='k')
	plt.axhline(1.0)
	set_xy_scales()
	set_xy_lims(-2000, 1000, 1e-2, 1e2)
	plt.xticks([-2000, -1000, 0, 1000])

def add_event_labels(mlight=50.0, delta=100.0, Tyrs=1.0, mT=100.0, gammab=1e-5, side='left'):
	plt.text(-1950, 8.8e9, r'$\beta \mathrm{\ decay\ background}$', c='#3F7BB6', fontsize=12, rotation=-6)
	plt.text(-1950, 0.05, r'$\mathrm{Const.\ background}\,\Gamma_\mathrm{b}$', c='k', fontsize=12, rotation=0)

	plt.text(-1900, 10**(8.1), r'$\Delta = ' + r'{}'.format(int(delta)) + r'\, \mathrm{meV}$', c='k', fontsize=12, rotation=0)
	plt.text(-1900, 10**(6.8), r'$m_\mathrm{lightest} = ' + r'{}'.format(int(mlight)) + r'\,\mathrm{meV}$', c='k', fontsize=12, rotation=0)

	plt.text(-1900, 10**(5.1), r'$T = ' + r'{}'.format(int(Tyrs)) + r'\, \mathrm{yr}$', c='k', fontsize=12, rotation=0)
	plt.text(-1900, 10**(3.8), r'$m_\mathrm{T} = ' + r'{}'.format(int(mT)) + r'\mathrm{g}$', c='k', fontsize=12, rotation=0)
	plt.text(-1900, 10**(2.5), r'$\Gamma_\mathrm{b} = 10^{' + r'{}'.format(int(np.log10(gammab))) + r'}\,\mathrm{Hz}$', c='k', fontsize=12, rotation=0)

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


if __name__ == '__main__':
	print('[utils.py] Starting utils.py')
