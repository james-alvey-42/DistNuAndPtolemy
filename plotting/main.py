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

def ptolemy_future(plot_dir='plots/', figname='ptolemy_future.pdf'):
	print('[main.py] Running: ptolemy_future')
	Tyrs, Delta, mT, Gammab, order, spin = 1., 100., 100., 1e-5, 'Normal', 'Dirac'
	
	plt.figure(figsize=(14,6))

	ax = plt.subplot(1, 2, 1)

	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy_summnu(data, order=order)
	plot_highp_DESI(order=order)
	plot_lowT_DESI(order=order)
	plot_LCDM_DESI(order=order)
	add_mlightest_zero(order=order)

	add_KATRIN(forecast=True)
	
	add_xy_labels(xlabel=r'$(1/3) \times \sum m_\nu \, \mathrm{[meV]}$')
	set_xy_lims(xmin=10)
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin, xmin=21.0)
	add_rate_axis(spin=spin)

	ax = plt.subplot(1, 2, 2)

	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy_summnu(data, order=order)
	plot_highp_DESI_no_detect(order='inverted')
	plot_lowT_DESI_no_detect(order='inverted')

	add_0vvb(forecast=True)
	
	add_xy_labels(xlabel=r'$(1/3) \times \sum m_\nu \, \mathrm{[meV]}$')
	set_xy_lims(xmin=10)
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order='Inverted', spin=spin)
	add_rate_axis(spin=spin)

	plt.tight_layout()
	print('[main.py] Saving:', plot_dir + figname)
	plt.savefig(plot_dir + figname)

def ptolemy_four_panel(plot_dir='plots/', figname='ptolemy_four_panel.pdf'):
	print('[main.py] Running: ptolemy_four_panel')
	fig = plt.figure(figsize=(12, 12))

	ax1 = plt.subplot(2, 2, 1)

	Tyrs, Delta, mT, Gammab, order, spin = 1.0, 20.0, 100.0, 1e-5, 'Normal', 'Dirac'
	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy(data)
	plot_highp_simple(order=order)
	plot_lowT_simple(order=order)
	plot_LCDM_simple(order=order)

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


if __name__ == '__main__':
	print('[main.py] Starting main.py')

	print('[main.py] Which plots to make?')
	rates = input('[main.py] Ptolemy rates (y/n): ')
	main = input('[main.py] Ptolemy main (y/n): ')
	future = input('[main.py] Ptolemy future (y/n): ')
	four = input('[main.py] Ptolemy four panel (y/n): ')
	if rates.lower()[0] == 'y':
		ptolemy_rates()
	if main.lower()[0] == 'y':
		ptolemy_main()
	if future.lower()[0] == 'y':
		ptolemy_future()
	if four.lower()[0] == 'y':
		ptolemy_four_panel()