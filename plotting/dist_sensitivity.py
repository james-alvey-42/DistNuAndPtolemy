import numpy as np
import matplotlib.pyplot as plt

data_dir = '../data/'

planck = np.loadtxt(data_dir + 'planck_sensitivity.txt')
cmbs4 = np.loadtxt(data_dir + 'cmb-s4_sensitivity.txt')
colors = ['black', 'green', 'purple', 'blue', 'red']

plt.figure(figsize=(25, 5))
labels = ['LCDM', 'High+DR', 'LowT+Mid', 'High', 'Low+DR']
for idx, col in enumerate(colors):
    ax = plt.subplot(1, 5, idx + 1)
    plt.plot(planck[:, 0], planck[:, idx + 1], c=col, lw=2.2, ls='-', label='Planck')
    plt.plot(cmbs4[:, 0], cmbs4[:, idx + 1], c=col, lw=2.2, ls=(1, (5, 1)), label='CMB-S4')
    plt.ylim(0, 5)
    plt.axvline(0.04)
    plt.xlabel(r'$m_\nu^{\Lambda\mathrm{CDM}}$')
    plt.ylabel(r'$\sqrt{-2 \log \mathcal{L}}$')
    plt.title(labels[idx])
    if idx == 0:
        plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('sensitivity.pdf')