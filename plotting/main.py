from utils import *
import numpy as np
import matplotlib.pyplot as plt

def ptolemy_main(plot_dir='plots/', figname='ptolemy_main.pdf'):
	Tyrs, Delta, mT, Gammab, order = 1, 100, 100, 1e-05, 'Normal'
	
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
	plt.savefig(plot_dir + figname)

def ptolemy_future(plot_dir='plots/', figname='ptolemy_future.pdf'):
	Tyrs, Delta, mT, Gammab, order, spin = 1, 100, 100, 1e-05, 'Normal', 'Dirac'
	
	plt.figure(figsize=(7,6))

	data = load_ptolemy(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin)

	plot_ptolemy_summnu(data, order=order)
	plot_highp_DESI(order=order)
	plot_lowT_DESI(order=order)
	plot_LCDM_DESI(order=order)

	add_KATRIN(forecast=True)
	
	add_xy_labels(xlabel=r'$(1/3) \times \sum m_\nu \, \mathrm{[meV]}$')
	set_xy_lims(xmin=20)
	add_case_labels(Tyrs=Tyrs, Delta=Delta, mT=mT, Gammab=Gammab, order=order, spin=spin, xmin=23.0)
	add_rate_axis(spin=spin)

	plt.tight_layout()
	plt.savefig(plot_dir + figname)


if __name__ == '__main__':
	print('Running main.py')

	ptolemy_main()
	ptolemy_future()