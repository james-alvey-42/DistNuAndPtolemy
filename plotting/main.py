from utils import *
import numpy as np
import matplotlib.pyplot as plt

def ptolemy_rates(plot_dir='plots/', figname='ptolemy_rates.pdf'):
	print('[main.py] Running: ptolemy_rates')
	f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [1, 1]}, figsize=(10, 6))

	delta, mlight, nloc = 100.0, 50.0, 10.0
	Tyrs, mT, gammab = 1, 100, 1e-5
	arr_dict = get_event_arrays(mlight=mlight, nloc=nloc)
	
	plt.sca(ax1)
	plot_all_events(arr_dict)
	add_event_labels(mlight=mlight, delta=delta, side='L')

	plt.text(-50.0, 1e12, r'$\mathbf{\Delta > m_\mathrm{lightest}}$', fontsize=14)
	plt.text(-100.0, 5e10, r'$m_\mathrm{lightest} = ' + r'{}'.format(int(mlight)) + r'\,\mathrm{meV}$', c='k', fontsize=12, rotation=0)

	xlabel, ylabel = None, r'$N_\mathrm{events}$'
	add_xy_labels(xlabel, ylabel, fontsize=16)
	ax1.xaxis.set_major_formatter(plt.NullFormatter())
	ax3.set_yticks([1e-2, 1e0, 1e2])


	plt.sca(ax3)

	plot_snr(arr_dict)
	xlabel, ylabel = r'$E_e - E_{\mathrm{end},0}\,\mathrm{[eV]}$', r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}}$'
	add_xy_labels(xlabel, ylabel, fontsize=16)
	ax3.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$'])
	ax3.set_yticks([1e-2, 1e0, 1e2])


	delta, mlight, nloc = 100.0, 500.0, 10.0
	arr_dict = get_event_arrays(mlight=mlight, nloc=nloc)

	plt.sca(ax2)

	plot_all_events(arr_dict)
	add_event_labels(mlight=mlight, delta=delta, side='R')

	plt.text(-50.0, 1e12, r'$\mathbf{\Delta < m_\mathrm{lightest}}$', fontsize=14)
	plt.text(-120.0, 5e10, r'$m_\mathrm{lightest} = ' + r'{}'.format(int(mlight)) + r'\,\mathrm{meV}$', c='k', fontsize=12, rotation=0)
	
	xlabel, ylabel = None, r'$N_\mathrm{events}$'
	add_xy_labels(xlabel, ylabel, fontsize=16)
	ax2.xaxis.set_major_formatter(plt.NullFormatter())


	plt.sca(ax4)

	plot_snr(arr_dict)
	xlabel, ylabel = r'$E_e - E_{\mathrm{end},0}\,\mathrm{[eV]}$', r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}}$'
	add_xy_labels(xlabel, ylabel, fontsize=16)
	ax4.set_yticks([1e-2, 1e0, 1e2])
	ax4.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$'])

	plt.suptitle(r'$\textbf{Experimental Configuration:\ }$' + r'$T = ' + r'{}'.format(int(Tyrs)) + r'\, \mathrm{yr},\ $' + r'$m_\mathrm{T} = ' + r'{}'.format(int(mT)) + r'\mathrm{g},\ $' + r'$\Gamma_\mathrm{b} = 7 \times 10^{' + r'{}'.format(int(np.log10(gammab)) - 2) + r'}\,\mathrm{Hz\ eV}^{-1},\ $' + r'$\Delta = ' + r'{}'.format(int(delta)) + r'\, \mathrm{meV}$', fontsize=14, 
		x=0.15, 
		y=.95, 
		horizontalalignment='left', 
		verticalalignment='top')
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.subplots_adjust(hspace=0.1)

	for ax in [ax1, ax2, ax3, ax4]:
		ax.tick_params(which='minor', length=3)
		ax.tick_params(which='major', length=5)

	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def ptolemy_main(plot_dir='plots/', figname='ptolemy_main.pdf'):
	print('[main.py] Running: ptolemy_main')
	Tyrs, Delta, mT, Gammab, order = 1., 100., 100., 1e-5, 'Normal'
	
	plt.figure(figsize=(14, 6))

	for idx, spin in enumerate(['Dirac', 'Majorana']):
		data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

		ax = plt.subplot(1, 2, idx + 1)
		plot_ptolemy(data)
		plot_highp(order=order)
		plot_lowT(order=order)
		plot_LCDM(order=order)

		add_KATRIN()
		if spin == 'Majorana':
			add_0vvb()
		
		add_xy_labels()
		set_xy_lims()
		add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
		add_rate_axis(spin=spin, labelpad=10 * idx + 20)
	
	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def ptolemy_binning_test(plot_dir='plots/', figname='ptolemy_binning_test.pdf'):
	print('[main.py] Running: ptolemy_binning_test')
	Tyrs, Delta, mT, Gammab, order, spin = 1., 100., 100., 1e-5, 'Normal', 'Dirac'
	
	plt.figure(figsize=(7, 6))

	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	ax = plt.subplot(1, 1, 1)
	plot_ptolemy(data)
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	plot_highp(order=order)
	plot_lowT(order=order)
	plot_LCDM(order=order)

	add_KATRIN()
	if spin == 'Majorana':
		add_0vvb()
	
	add_xy_labels()
	set_xy_lims()
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	add_rate_axis(spin=spin)
	
	Gammab = 2e-6
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	plot_ptolemy(data, overlay=True)

	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def ptolemy_future(plot_dir='plots/', figname='ptolemy_future.pdf'):
	print('[main.py] Running: ptolemy_future')
	Tyrs, Delta, mT, Gammab, order, spin = 1., 100., 100., 1e-5, 'Normal', 'Dirac'
	
	plt.figure(figsize=(14,6))

	ax = plt.subplot(1, 2, 1)
	plt.title(r'$\mathrm{DESI/EUCLID\ Detection}$', fontsize=22)

	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy_summnu(data, order=order)
	plot_highp_DESI(order=order)
	plot_lowT_DESI(order=order)
	plot_LCDM_DESI(order=order)
	add_mlightest_zero(order=order)

	add_KATRIN(forecast=True)
	
	add_xy_labels(xlabel=r'$(1/3) \times \sum m_\nu \, \mathrm{[meV]}$')
	set_xy_lims(xmin=5)
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin, xmin=5.5)
	add_rate_axis(spin=spin)

	ax = plt.subplot(1, 2, 2)
	plt.title(r'$\mathrm{No\ DESI/EUCLID\ Detection}$', fontsize=22)

	order, spin = 'Inverted', 'Majorana'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy_summnu(data, order=order)
	plot_highp_DESI_no_detect(order=order)
	plot_lowT_DESI_no_detect(order=order)
	plot_LCDM_DESI_no_detect(order=order)
	add_mlightest_zero(order=order)

	add_0vvb(forecast=True)
	add_KATRIN(forecast=True)
	
	add_xy_labels(xlabel=r'$(1/3) \times \sum m_\nu \, \mathrm{[meV]}$')
	set_xy_lims(xmin=5)
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin, xmin=5.5)
	add_rate_axis(spin=spin, labelpad=30)

	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def ptolemy_four_panel(plot_dir='plots/', figname='ptolemy_four_panel.pdf'):
	print('[main.py] Running: ptolemy_four_panel')
	fig = plt.figure(figsize=(13, 11))
	Tyrs, Delta, mT, Gammab, order, spin = 1., 100., 100., 1e-5, 'Normal', 'Dirac'
	overlay_data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	ax1 = plt.subplot(2, 2, 1)

	Tyrs, Delta, mT, Gammab, order, spin = 1.0, 20.0, 100.0, 1e-5, 'Normal', 'Dirac'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy(data)
	plot_highp_simple(order=order)
	plot_lowT_simple(order=order)
	plot_LCDM_simple(order=order)
	plot_ptolemy(overlay_data, overlay=True)

	add_KATRIN()
	if spin == 'Majorana':
		add_0vvb()
	
	add_xy_labels()
	set_xy_lims()
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	add_rate_axis(spin=spin, labelpad=20, nolabel=True)
	plt.title(r'$\mathrm{5x\ Smaller\ Resolution,\ }\Delta$', fontsize=20)

	ax2 = plt.subplot(2, 2, 2)
	Tyrs, Delta, mT, Gammab, order, spin = 1.0, 100.0, 50.0, 1e-5, 'Normal', 'Dirac'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy(data)
	plot_highp_simple(order=order)
	plot_lowT_simple(order=order)
	plot_LCDM_simple(order=order)
	plot_ptolemy(overlay_data, overlay=True)

	add_KATRIN()
	if spin == 'Majorana':
		add_0vvb()
	
	add_xy_labels(ylabel=None)
	set_xy_lims()
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	add_rate_axis(spin=spin, labelpad=30)
	plt.title(r'$\mathrm{2x\ Smaller\ Sample,\ }M_\mathrm{T}$', fontsize=20)

	ax3 = plt.subplot(2, 2, 3)
	Tyrs, Delta, mT, Gammab, order, spin = 1.0, 100.0, 100.0, 1e-4, 'Normal', 'Dirac'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy(data)
	plot_highp_simple(order=order)
	plot_lowT_simple(order=order)
	plot_LCDM_simple(order=order)
	plot_ptolemy(overlay_data, overlay=True)

	add_KATRIN()
	if spin == 'Majorana':
		add_0vvb()
	
	add_xy_labels()
	set_xy_lims()
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	add_rate_axis(spin=spin, labelpad=20, nolabel=True)
	plt.title(r'$\mathrm{10x\ Larger\ Background,\ }\Gamma_\mathrm{b}$', fontsize=20)

	ax4 = plt.subplot(2, 2, 4)
	Tyrs, Delta, mT, Gammab, order, spin = 3.0, 100.0, 100.0, 1e-5, 'Normal', 'Dirac'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy(data)
	plot_highp_simple(order=order)
	plot_lowT_simple(order=order)
	plot_LCDM_simple(order=order)
	plot_ptolemy(overlay_data, overlay=True)

	add_KATRIN()
	if spin == 'Majorana':
		add_0vvb()
	
	add_xy_labels(ylabel=None)
	set_xy_lims()
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)
	add_rate_axis(spin=spin, labelpad=30)
	plt.title(r'$\mathrm{3x\ Longer\ Exposure,\ }T$', fontsize=20)

	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_densities(plot_dir='plots/', figname='cosmo_densities.pdf'):
	c1, c2, c3 = '#98065a', '#2f6e1a', '#0c0c0c' # purple, green, black
	c1, c2, c3 = '#D84797', '#2f6e1a', '#0c0c0c' # purple, green, black
	# c1 = '#f75f00'
	a, nu, de, b, c, gamma = np.loadtxt('../data/densities/a.dat'), np.loadtxt('../data/densities/nu_0.12.dat'), np.loadtxt('../data/densities/lambda_0.12.dat'), np.loadtxt('../data/densities/b_0.12.dat'), np.loadtxt('../data/densities/c_0.12.dat'), np.loadtxt('../data/densities/gamma_0.12.dat')
	nu_m = np.loadtxt('../data/densities/nu_0.0.dat')
	plt.figure(figsize=(8, 5))
	plt.loglog(a, nu_m, c=cosmo_color('HEDR'), lw=1.5, ls=(1, (5, 1)))
	plt.loglog(a, nu, c=cosmo_color('HEDR'), lw=2.2)
	plt.loglog(a, de, c='k', lw=1.0, zorder=1)
	plt.loglog(a, b, c='k', lw=1.0, zorder=1)
	plt.loglog(a, c, c='k', lw=1.0, zorder=1)
	plt.loglog(a, gamma, c='k', lw=1.0, zorder=1)
	plt.xlabel(r'$\mathrm{Scale\,\,Factor}\,\,a/a_0 = (1 + z)^{-1}$', fontsize=16)
	plt.ylabel(r'$\mathrm{Energy\,\,Density}\,\,\Omega_{i}(a)$', fontsize=16)
	plt.xlim(1e-6, 1e0)
	plt.ylim(1e-4, 2e0)
	
	plt.text(1.5e-6, 5.8e-3, r'$\mathrm{Cold\,\,Dark\,\,Matter}$', rotation=40, fontsize=12)
	plt.text(4e-6, 2.8e-3, r'$\mathrm{Baryons}$', rotation=40, fontsize=12)
	plt.text(2.8e-2, 1.5e-4, r'$\mathrm{Dark\,\,Energy}$', rotation=68, fontsize=12)
	plt.text(1.8e-6, 6.5e-1, r'$\mathrm{Photons}$', rotation=0, fontsize=12)
	plt.text(1.5e-6, 2.8e-1, r'$\mathbf{Neutrinos}$', rotation=0, fontsize=12, color=cosmo_color('HEDR'))
	plt.text(1.7e-2, 1.7e-3, r'$\mathrm{Massless}$', rotation=-40, fontsize=12, color=cosmo_color('HEDR'))
	plt.text(1.7e-2, 1.3e-2, r'$\Sigma m_\nu = 0.12\,\mathrm{eV}$', rotation=-3, fontsize=12, color=cosmo_color('HEDR'))

	plt.text(2.1e-4, 1.6e-4, r'$\mathrm{Matter-radiation\,\,eq.,\,\,}z_{\mathrm{eq}}$', rotation=90, fontsize=12, color='gray')
	plt.text(6.9e-4, 1.6e-4, r'$\mathrm{Photon\,\,decoupling,\,\,}z_{\mathrm{dec}}$', rotation=90, fontsize=12, color='gray')
	plt.text(0.95e-2, 1.6e-4, r"$\nu\mathrm{'s\,\,non-rel.,\,\,}z_{\mathrm{NR}}$", rotation=90, fontsize=12, color='gray')
	plt.axvline((1 + 3402)**(-1), c='gray', ls='-', lw=1.0, alpha=0.7, zorder=0)
	plt.axvline((1 + 1089)**(-1), c='gray', ls='-', lw=1.0, alpha=0.7, zorder=0)
	plt.axvline((1 + 76)**(-1), c='gray', ls='-', lw=1.0, alpha=0.7, zorder=0)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_distributions(plot_dir='plots/', figname='cosmo_distributions.pdf'):
	fig = plt.figure(figsize=(14, 12))
	
	ax = plt.subplot(2, 2, 1)

	plot_distributions()

	set_xy_lims(xmin=1e-1, xmax=1e2, ymin=0., ymax=50.)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$q_\nu \equiv p_\nu/T_{\nu, 0}$")
	ax.set_ylabel(r"$\frac{\mathrm{d}\rho^\mathrm{NR}_\nu}{\mathrm{d} \log q_\nu} \, \mathrm{[eV\ cm}^{-3}\mathrm{]}$")

	add_cosmo_cases()

	ax = plt.subplot(2, 2, 2)

	plot_energy_evolution()
	set_xy_lims(xmin=1e0, xmax=1e4, ymin=0.95, ymax=1.30)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$z$")
	ax.set_ylabel(r"$(\rho_\nu+\rho_{\rm DR})/\rho_\nu|_{\Lambda {\rm CDM}}$")

	ax = plt.subplot(2, 2, 3)

	plot_distribution_Cl()
	set_xy_lims(xmin=2., xmax=2500, ymin=-0.004, ymax=0.004)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$\ell$")
	ax.set_ylabel(r"$(C^{\rm TT}_\ell-C^{\rm TT}_\ell|_{\Lambda {\rm CDM}})/C^{\rm TT}_\ell|_{\Lambda {\rm CDM}}$")

	ax = plt.subplot(2, 2, 4)

	plot_rho_hist(case='LCDM')
	plot_rho_hist(case='LEDR')
	plot_rho_hist(case='HE', nratio=0.1)
	plot_rho_hist(case='HEDR', nratio=0.1)
	plot_rho_hist(case='LTM')
	set_xy_lims(xmin=0., xmax=50., ymin=0., ymax=1.)
	set_xy_scales(xscale='linear', yscale='linear')

	ax.set_xlabel(r"$\rho_{\nu,0}^\mathrm{NR}\,\mathrm{[eV\ cm}^{-3}\mathrm{]}$")
	ax.set_ylabel(r"$\mathrm{Posterior\ Density}$")

	plt.text(0.13, 0.92, r"$\textbf{Data:\ }\mathrm{Planck\ TTTEEE+lowE\ +\ BAO}$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)
	plt.text(0.13, 0.83, r'$\rho_{\nu,0}^\mathrm{NR} = (\sum m_\nu) \times 2 T_{\nu,0}^3 \int{\frac{\mathrm{d} q_\nu}{2\pi^2} \, q_\nu^2 f_\nu(q_\nu)}$',
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k')

	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_appendix_nu(plot_dir='plots/', figname='cosmo_appendix_nu.pdf'):
	fig = plt.figure(figsize=(18, 6))

	ax = plt.subplot(1, 3, 1)
	plot_distribution_Cl(lensed=False)
	set_xy_lims(xmin=2., xmax=2500, ymin=-0.004, ymax=0.004)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$\ell$", fontsize=22)
	ax.set_ylabel(r"$(C^{\rm TT}_\ell-C^{\rm TT}_\ell|_{\Lambda {\rm CDM}})/C^{\rm TT}_\ell|_{\Lambda {\rm CDM}}$", fontsize=22)
	plt.text(0.7, 0.92, 'Unlensed',
		transform=plt.gca().transAxes,
		fontsize=18,
		color='k')
	plt.text(0.05, 0.035,r"$f_\nu = \Omega_\nu / \Omega_{\rm m} = 0.009$, $N_{\rm eff} = 3.044$",
		transform=plt.gca().transAxes,
		fontsize=16,
		color='k',
		rotation=0)
	
	ax = plt.subplot(1, 3, 2)

	plot_energy_evolution(nu_only=True)
	set_xy_lims(xmin=1e0, xmax=1e4, ymin=0., ymax=1.1)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$z$", fontsize=22)
	ax.set_ylabel(r"$\rho_\nu/\rho_\nu|_{\Lambda {\rm CDM}}$", fontsize=22)

	ax = plt.subplot(1, 3, 3)
	plot_eos()
	set_xy_lims(xmin=1e0, xmax=1e4, ymin=0., ymax=1.1)
	set_xy_scales(xscale='log', yscale='linear')

	ax.set_xlabel(r"$z$", fontsize=22)
	ax.set_ylabel(r"$3 P_\nu / \rho_\nu$", fontsize=22)
	plt.text(0.05, 0.92, 'Eq. of State',
		transform=plt.gca().transAxes,
		fontsize=18,
		color='k')

	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_cmb_sensitivity(plot_dir='plots/', figname='cosmo_cmb_sensitivity.pdf', two_panel=True):
	if two_panel:
		fig = plt.figure(figsize=(11, 5))
	else:
		fig = plt.figure(figsize=(5, 5))
	cases = ['LEDR', 'HE', 'HEDR', 'LTM']
	
	if two_panel:
		ax = plt.subplot(1, 2, 1)
	else:
		ax = plt.subplot(1, 1, 1)

	for case in cases:
		plot_CMB_sensitivity(case, expt='planck', ls='-', alpha=1.0)

	plt.text(0.05, 0.9, r'$\textbf{Planck}$',
		transform=plt.gca().transAxes,
		fontsize=16,
		c=cosmo_color('LCDM'))
	ax.set_xlabel(r"$\rho_{\nu, 0}^\mathrm{NR} \, \mathrm{[eV\ cm}^{-3}\mathrm{]}$", fontsize=16)
	ax.set_ylabel(r"$\sqrt{\Delta \chi^2} = (\sqrt{-2 \log\mathcal{L}})_\mathrm{optimal}$", fontsize=16)
	plt.xlim(0., 40.)
	plt.ylim(0., 5.)
	plt.axvline(14., c='k', ls='-', lw=1.0, zorder=-5)
	
	plt.text(0.75,0.40,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LEDR'),
		rotation=40, alpha=1.0)
	plt.text(0.75,0.04, r"\textbf{H}\boldmath{$\nu$}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('HE'),
		rotation=10, alpha=1.0)
	plt.text(0.75,0.13,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('HEDR'),
		rotation=15, alpha=1.0)
	plt.text(0.55,0.035,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LTM'),
		rotation=3, alpha=1.0)
	plt.annotate(s='', xytext=(14.5, 3.5), xy=(25.0, 3.5), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': 'k'}, color='k')
	plt.fill_betweenx([0., 5.], 14, 40, 
		color=cosmo_color('LCDM'), alpha=0.1)
	plt.text(15, 3.2, r'$\mathrm{Excluded\ by}$',
		fontsize=12)
	plt.text(15, 2.9, r'$\mathrm{Planck\ 2018}$',
		fontsize=12)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	if two_panel:
		ax = plt.subplot(1, 2, 2)

	for case in cases:
		plot_CMB_sensitivity(case, expt='cmbs4')

	plt.text(0.05, 0.9, r'$\textbf{CMB-S4}$',
		transform=plt.gca().transAxes,
		fontsize=16,
		c=cosmo_color('LCDM'))
	ax.set_xlabel(r"$\rho_{\nu, 0}^\mathrm{NR} \, \mathrm{[eV\ cm}^{-3}\mathrm{]}$", fontsize=16)
	ax.set_ylabel(r"$\sqrt{\Delta \chi^2} = (\sqrt{-2 \log\mathcal{L}})_\mathrm{optimal}$", fontsize=16)
	plt.xlim(0., 40.)
	plt.ylim(0., 5.)
	plt.axvline(14., c='k', ls='-', lw=1.0, zorder=-5)

	plt.text(0.11,0.40,r"\textbf{L}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LEDR'),
		rotation=70)
	plt.text(0.75,0.26, r"\textbf{H}\boldmath{$\nu$}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('HE'),
		rotation=14)
	plt.text(0.75,0.61,r"\textbf{H}\boldmath{$\nu$}\textbf{-DR}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('HEDR'),
		rotation=42)
	plt.text(0.55,0.1,r"\textbf{LT+Mid}",transform=plt.gca().transAxes,
		fontsize=10,
		color=cosmo_color('LTM'),
		rotation=9)
	plt.annotate(s='', xytext=(14.5, 3.5), xy=(25.0, 3.5), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': 'k'}, color='k')
	plt.fill_betweenx([0., 5.], 14, 40, 
		color=cosmo_color('LCDM'), alpha=0.1)
	plt.text(15, 3.2, r'$\mathrm{Excluded\ by}$',
		fontsize=12)
	plt.text(15, 2.9, r'$\mathrm{Planck\ 2018}$',
		fontsize=12)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_contour(plot_dir='plots/', figname='cosmo_contour.pdf'):
	plt.figure(figsize=(15, 15))
	plot_main_contour()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_mpk(plot_dir='plots/', figname='cosmo_mpk.pdf'):
	plt.figure(figsize=(7, 6))
	ax = plt.subplot(1, 1, 1)
	plot_distribution_Pk()
	plt.axvline(0.1, c='k', ls=(1, (5, 1)), lw=1.2, zorder=-10)
	plt.annotate(s='', xytext=(0.11, 0.023), xy=(0.25, 0.023), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': 'k'}, color='k')
	plt.text(0.11, 0.02, r'$\mathrm{Non-linear}$', fontsize=12)
	# ax = plt.subplot(1, 2, 2)
	# plot_distribution_Pk_ratio()
	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_appendix_contour(plot_dir='plots/', figname='cosmo_contour_app.pdf'):
	plt.figure(figsize=(20, 20))
	plot_appendix_contour()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

if __name__ == '__main__':
	print('[main.py] Starting main.py')
	
	paper = input('[main.py] Paper (Ptolemy/Cosmo): ')
	if paper.lower()[0] == 'p':
		rates = input('[main.py] Ptolemy rates (y/n): ')
		main = input('[main.py] Ptolemy main (y/n): ')
		# binning = input('[main.py] Ptolemy binning (y/n): ')
		future = input('[main.py] Ptolemy future (y/n): ')
		four = input('[main.py] Ptolemy four panel (y/n): ')
		if rates.lower()[0] == 'y':
			ptolemy_rates()
		if main.lower()[0] == 'y':
			ptolemy_main()
		# if binning.lower()[0] == 'y':
		# 	ptolemy_binning_test()
		if future.lower()[0] == 'y':
			ptolemy_future()
		if four.lower()[0] == 'y':
			ptolemy_four_panel()
	
	elif paper.lower()[0] == 'c':
		density = input('[main.py] Densities (y/n): ')
		dists = input('[main.py] Distributions (y/n): ')
		cmb = input('[main.py] CMB Sensitivity (y/n): ')
		contour = input('[main.py] Countour Plot (y/n): ')
		mpk = input('[main.py] Matter power spectrum (y/n): ')
		appcontour = input('[main.py] Appendix contour (y/n): ')
		nuonly = input('[main.py] Nu Appendix (y/n): ')
		if density.lower()[0] == 'y':
			cosmo_densities()
		if dists.lower()[0] == 'y':
			cosmo_distributions()
		if cmb.lower()[0] == 'y':
			cosmo_cmb_sensitivity()
		if contour.lower()[0] == 'y':
			cosmo_contour()
		if mpk.lower()[0] == 'y':
			cosmo_mpk()
		if appcontour.lower()[0] == 'y':
			cosmo_appendix_contour()
		if nuonly.lower()[0] == 'y':
			cosmo_appendix_nu()