import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.special import zeta

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

def load_ptolemy(Tyrs=1, Delta=100, mT=100, Gammab=1e-5, order='normal', spin='Dirac', filename=None, data_dir='../data/'):
	if filename is None:
		filename = '[t]{}_[d]{}_[m]{}_[o]{}_[s]{}_[b]{}.txt'.format(Tyrs, Delta, mT, order.lower()[0], spin.lower()[0], Gammab)
	try:
		data = np.loadtxt(data_dir + filename)
		return data
	except OSError:
		print('(ERROR) Cannot find file:', data_dir + filename)

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
	plt.text(191.0, 240.0, 'PTOLEMY', fontsize=14, color='lightgreen')
	plt.xscale('log')
	plt.yscale('log')

def add_case_labels(Tyrs=1, Delta=100, mT=100, Gammab=1e-5, order='Normal', spin='Dirac', xmin=12.0):
	plt.text(xmin, 5.4, r'$\mathrm{' + r'{}'.format(order) + r'\ ordering}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 3.8, r'$\mathrm{' + r'{}'.format(spin) + r'}\ \nu$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 2.6, r'$T = ' + r'{}'.format(Tyrs) + r'\, \mathrm{yr}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.8, r'$\Delta = ' + r'{}'.format(Delta) + r'\, \mathrm{meV}$', fontsize=12, color='k', rotation=0)
	plt.text(xmin, 1.3, r'$m_{{}^{3}\mathrm{H}} ='  + r'{}'.format(mT) + r'\ \mathrm{g}$', fontsize=12, color='k', rotation=0)

def add_rate_axis(spin='Dirac', labelpad=20):
	if 'maj' in spin.lower():
		cfactor = 2.0
	else:
		cfactor = 1.0
	
	ax = plt.gca()
	twinax = plt.gca().twinx()
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
	plt.text(325.0, 3.0, r'$\langle p_\nu\rangle \gg 3T$', fontsize=10, rotation=-34, color="#3F7BB6")

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
	plt.text(191.0, 240.0, 'PTOLEMY', fontsize=14, color='lightgreen')
	plt.xscale('log')
	plt.yscale('log')

def plot_highp_DESI(order='normal'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.04 / sum_mnu_arr)
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, n_upper_arr, color='#3F7BB6', lw=2.2, zorder=-9, alpha=1.)

	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, n_lower_arr, n_upper_arr, facecolor="#3F7BB6", color="#3F7BB6", edgecolor="#d9e5f0", linewidth=0.0, zorder=-9, alpha=0.2)
	plt.text(92.0, 3.65, r'$\langle p_\nu\rangle \gg 3T$', fontsize=10, rotation=-28, color="#3F7BB6")

def plot_lowT_DESI(order='normal', data_dir='../data/'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	n_upper_arr = n_FD() * (0.08 / sum_mnu_arr)
	n_lower_arr = n_FD() * (0.04 / sum_mnu_arr)
	Tnu_upper_arr = 1.95 * np.power(n_upper_arr/n_FD(), 1./3.)
	Tnu_lower_arr = 1.95 * np.power(n_lower_arr/n_FD(), 1./3.)
	
	clus_data_mlight, clus_data_delta, _ = np.loadtxt(data_dir + 'lowT_clustering_008.txt', unpack=True)
	delta_fn_upper = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	clus_data_mlight, clus_data_delta, _ = np.loadtxt(data_dir + 'lowT_clustering_004.txt', unpack=True)
	delta_fn_lower = interp1d(1e3 * clus_data_mlight, clus_data_delta, kind='cubic', fill_value='extrapolate')
	
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_upper(mlight_arr)) * n_FD(Tnu_upper_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.plot((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower(mlight_arr)) * n_FD(Tnu_lower_arr), color='purple', lw=2.2, zorder=-9, alpha=1.)
	plt.fill_between((1./3.) * 1e3 * sum_mnu_arr, (1 + delta_fn_lower(mlight_arr)) * n_FD(Tnu_lower_arr), (1 + delta_fn_upper(mlight_arr)) * n_FD(Tnu_upper_arr), facecolor="purple", color="purple", edgecolor="#dfd1eb", linewidth=0.0, zorder=-9, alpha=0.2)
	plt.text(63, 19.0, r'$T_{\nu, 0} < 1.95 \, \mathrm{K}$', fontsize=10, rotation=-23, color="purple")
	plt.text(34.0, 13.4, 'DESI/EUCLID', fontsize=12, rotation=-29, color='#faf9fc')

def plot_LCDM_DESI(order='normal'):
	mlight_arr = np.geomspace(0.001, 1000, 1000)
	sum_mnu_arr = sum_mnu(1e-3 * mlight_arr, order=order)
	LCDM_pos = np.argmin(np.abs(sum_mnu_arr - 0.08))
	plt.plot((1./3.) * 1e3 * sum_mnu_arr[:LCDM_pos], (1 + delta_LCDM(1e-3 * mlight_arr[:LCDM_pos])) * n_FD(), color='darkgoldenrod', lw=2.2, zorder=-9, alpha=1.0)
	plt.scatter([80./3.], [n_FD() * (1 + delta_LCDM(1e-3 * mlight_arr[LCDM_pos]))], 
			marker='|',
			c='darkgoldenrod',
			alpha=1.0, 
			s=100,
			linewidths=0.1,
			edgecolors='k',
			zorder=9
			)
	plt.text(28., 40.0, r'$T_{\nu, 0} = 1.95 \, \mathrm{K}$', fontsize=10, rotation=-29, color="darkgoldenrod")

if __name__ == '__main__':
	print('Running utils.py')