from utils import *
import numpy as np
import matplotlib.pyplot as plt

def ptolemy_rates(plot_dir='plots/', figname='ptolemy_rates.pdf'):
	print('[main.py] Running: ptolemy_rates')
	f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [1, 1]}, figsize=(10, 6))

	delta, mlight, nloc = 100.0, 50.0, 10.0
	arr_dict = get_event_arrays(mlight=mlight, nloc=nloc)
	
	plt.sca(ax1)
	plot_all_events(arr_dict)
	add_event_labels(mlight=mlight, delta=delta, side='L')

	plt.text(-50.0, 1e12, r'$\mathbf{\Delta > m_\mathrm{lightest}}$', fontsize=14)

	xlabel, ylabel = None, r'$N_\mathrm{events}$'
	add_xy_labels(xlabel, ylabel)
	ax1.xaxis.set_major_formatter(plt.NullFormatter())


	plt.sca(ax3)

	plot_snr(arr_dict)
	xlabel, ylabel = r'$E_e - E_{\mathrm{end},0}\,\mathrm{[eV]}$', r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}}$'
	add_xy_labels(xlabel, ylabel)
	ax3.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$'])


	delta, mlight, nloc = 100.0, 500.0, 10.0
	arr_dict = get_event_arrays(mlight=mlight, nloc=nloc)

	plt.sca(ax2)

	plot_all_events(arr_dict)
	add_event_labels(mlight=mlight, delta=delta, side='R')

	plt.text(-50.0, 1e12, r'$\mathbf{\Delta < m_\mathrm{lightest}}$', fontsize=14)
	
	xlabel, ylabel = None, r'$N_\mathrm{events}$'
	add_xy_labels(xlabel, ylabel)
	ax2.xaxis.set_major_formatter(plt.NullFormatter())


	plt.sca(ax4)

	plot_snr(arr_dict)
	xlabel, ylabel = r'$E_e - E_{\mathrm{end},0}\,\mathrm{[eV]}$', r'$N_\mathrm{sig.}/\sqrt{N_\mathrm{back.}}$'
	add_xy_labels(xlabel, ylabel)
	ax4.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$'])

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1)

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
	plt.title(r'$\mathrm{(a)\ DESI/EUCLID\ Detection}$', fontsize=16)

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
	plt.title(r'$\mathrm{(b)\ No\ DESI/EUCLID\ Detection}$', fontsize=16)

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
	fig = plt.figure(figsize=(12, 12))
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

	plt.text(0.18, 0.92, r"$\textbf{Data:\ }\mathrm{Planck\ TTTEEE+lowE+lowl\ +\ BAO}$",
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k',
		rotation=0)
	plt.text(0.18, 0.83, r'$\rho_{\nu,0}^\mathrm{NR} = (\sum m_\nu) \times 2 T_{\nu,0}^3 \int{\frac{\mathrm{d}\log q_\nu}{2\pi^2} \, q_\nu^3 f_\nu(q_\nu)}$',
		transform=plt.gca().transAxes,
		fontsize=14,
		color='k')

	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def cosmo_cmb_sensitivity(plot_dir='plots/', figname='cosmo_cmb_sensitivity.pdf'):
	fig = plt.figure(figsize=(12, 6))
	cases = ['LEDR', 'HE', 'HEDR', 'LTM']
	
	ax = plt.subplot(1, 2, 1)

	for case in cases:
		plot_CMB_sensitivity(case, expt='planck')

	ax.set_xlabel(r"$\rho_{\nu, 0}^\mathrm{NR} \, \mathrm{[eV\ cm}^{-3}\mathrm{]}$")
	ax.set_ylabel(r"$\sqrt{-2 \log\mathcal{L}}$")
	plt.xlim(0., 50.)
	plt.ylim(0., 5.)
	plt.axvline(14., c='k', ls=(1, (5, 1)), lw=1.8, zorder=-5)

	ax = plt.subplot(1, 2, 2)

	for case in cases:
		plot_CMB_sensitivity(case, expt='cmbs4')

	ax.set_xlabel(r"$\rho_{\nu, 0}^\mathrm{NR} \, \mathrm{[eV\ cm}^{-3}\mathrm{]}$")
	ax.set_ylabel(r"$\sqrt{-2 \log\mathcal{L}}$")
	plt.xlim(0., 50.)
	plt.ylim(0., 5.)
	plt.axvline(14., c='k', ls=(1, (5, 1)), lw=1.8, zorder=-5)

	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

if __name__ == '__main__':
	print('[main.py] Starting main.py')

	paper = input('[main.py] Paper (Ptolemy/Cosmo): ')
	if paper.lower()[0] == 'p':
		rates = input('[main.py] Ptolemy rates (y/n): ')
		main = input('[main.py] Ptolemy main (y/n): ')
		binning = input('[main.py] Ptolemy binning (y/n): ')
		future = input('[main.py] Ptolemy future (y/n): ')
		four = input('[main.py] Ptolemy four panel (y/n): ')
		if rates.lower()[0] == 'y':
			ptolemy_rates()
		if main.lower()[0] == 'y':
			ptolemy_main()
		if binning.lower()[0] == 'y':
			ptolemy_binning_test()
		if future.lower()[0] == 'y':
			ptolemy_future()
		if four.lower()[0] == 'y':
			ptolemy_four_panel()
	
	elif paper.lower()[0] == 'c':
		dists = input('[main.py] Distributions (y/n): ')
		cmb = input('[main.py] CMB Sensitivity (y/n): ')
		if dists.lower()[0] == 'y':
			cosmo_distributions()
		if cmb.lower()[0] == 'y':
			cosmo_cmb_sensitivity()