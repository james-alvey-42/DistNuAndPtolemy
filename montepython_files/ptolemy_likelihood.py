import numpy as np
from scipy.special import gamma
import os

import functools

class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

class ptolemy(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(50)

        self.mlightest_fid = data.mcmc_parameters['mlightest_fid']['current']
        self.delta = data.mcmc_parameters['delta']['current']

        self.Ei_arr = np.linspace(self.Eend0() - 5000., self.Eend0() + 10000., int(15000/self.delta))
        self.order = 'normal'
        self.Tyrs = data.mcmc_parameters['Tyrs']['current']
        self.n0 = 14e3 / (np.sum(self.masses(self.mlightest_fid, order="normal"))) / 2
        self.NT = 1.9972819100287977e+25
        self.gamma_b = 1e-5
        self.Nt_arr = self.N_total(self.Ei_arr, self.Tyrs, self.delta, self.mlightest_fid, self.n0, self.NT, order=self.order, DEEnd=0.0, gamma_b=self.gamma_b)

        self.border = 20
        self.ln_Nt_factorial = np.zeros(len(self.Nt_arr))
        self.ln_Nt_factorial[self.Nt_arr < self.border] = np.log(gamma(self.Nt_arr[self.Nt_arr < self.border] + 1))
        self.ln_Nt_factorial[self.Nt_arr > self.border] = self.Nt_arr[self.Nt_arr > self.border] * np.log(self.Nt_arr[self.Nt_arr > self.border]) - self.Nt_arr[self.Nt_arr > self.border] + np.log(1./30 + self.Nt_arr[self.Nt_arr > self.border] * (1 + 4 * self.Nt_arr[self.Nt_arr > self.border] * (1 + 2 * self.Nt_arr[self.Nt_arr > self.border]))) / 6. + 0.5 * np.log(np.pi)

        return

    @vectorize
    def integrator(self, f, a, b):
        sub = (b - a) / 2.
        add = (b + a) / 2.
        if sub == 0:
            return 0.
        return sub * np.dot(f(sub * self.points + add), self.weights)

    def F(self, Ee):
        '''
        Equation 3.6 - Fermi function F(Z, Ee) with Z = 2
        '''
        me = 511 * 1e6 # meV
        pe = np.sqrt(Ee**2 - me**2)
        Z = 2
        alpha = 1 / 137.036
        eta = Z * alpha * Ee / pe
        return 2 * np.pi * eta / (1 - np.exp(- 2 * np.pi * eta))

    def sigmav(self, Ee):
        '''
        Equation 3.5 - cross-section multiplied by v_\nu
        '''
        GF = 1.166 * 1e-29 # meV^-2
        m3He = 2808.391 # MeV
        m3H = 2808.921 # MeV
        me = 511 * 1e6 # meV
        pe = np.sqrt(Ee**2 - me**2)
        F2 = 0.9987
        GT2 = 2.788
        gA = 1.2695 # Footnote 5
        return (GF**2 / (2 * np.pi)) * self.F(Ee) * (m3He/m3H) * Ee * pe * (F2 + gA**2 * GT2)

    def Ue(self, order='normal'):
        '''
        Equation 3.7 - Mixing angles
        '''
        s12sq = 0.32
        if order == 'normal':
            s13sq = 2.16 * 1e-2
        else:
            s13sq = 2.22 * 1e-2
        c12sq = 1 - s12sq
        c13sq = 1 - s13sq
        return c12sq * c13sq, s12sq * c13sq, s13sq

    def fc(self, mi_meV, dist="Gaussian"):
        '''
        Equation 2.3 - gravitational clustering factors
        '''
        if dist == "Gaussian":
            return 0.
        return 76.5 * (mi_meV / 1000)**(2.21)

    def GammaCNB(self, m1, m2, m3, Ee, n0, NT, order='normal', total=True):
        Ue1sq, Ue2sq, Ue3sq = self.Ue(order)
        Gamma1 = NT * self.sigmav(Ee) * n0 * Ue1sq * (1 + self.fc(m1))
        Gamma2 = NT * self.sigmav(Ee) * n0 * Ue2sq * (1 + self.fc(m2))
        Gamma3 = NT * self.sigmav(Ee) * n0 * Ue3sq * (1 + self.fc(m3))
        if total:
            return Gamma1 + Gamma2 + Gamma3
        else:
            return Gamma1, Gamma2, Gamma3

    def mbetasq(self, m1, m2, m3, order='normal'):
        '''
        Equation 2.1 - effective neutrino beta decay mass
        '''
        Ue1sq, Ue2sq, Ue3sq = self.Ue(order)
        return Ue1sq * m1**2 + Ue2sq * m2**2 + Ue3sq * m3**2

    def Eend0(self):
        '''
        From 0706.0897 (eq. 2)
        '''
        m3He = 2808.391 * 1e9 # meV
        m3H = 2808.921 * 1e9 # meV
        me = 511 * 1e6 # meV
        return (1/(2 * m3H)) * (m3H**2 + me**2 - m3He**2)

    def H(self, Ee, mi, DEEnd=0.0):
        '''
        Equation 3.9 - H(Ee, mi) function
        '''
        y = np.heaviside(self.Eend0() + DEEnd - Ee - mi, 0) * (self.Eend0() + DEEnd- Ee - mi)
        m3He = 2808.391 * 1e9 # meV
        m3H = 2808.921 * 1e9 # meV
        me = 511 * 1e6 # meV
        return (1 - me**2/(Ee * m3H)) * np.sqrt(y * (y + (2 * mi * m3He)/(m3H))) * np.power(1 - 2 * Ee / m3H + me**2 / m3H**2, -2.0) * (y + (mi/m3H)*(m3He + mi))

    def dGbdEe(self, m1, m2, m3, Ee, NT, order='normal', DEEnd=0.0):
        '''
        Equation 3.8 - Beta decay rate
        '''
        me = 511 * 1e6 # meV
        pe = np.sqrt(Ee**2 - me**2)
        pnu = pe
        Enu = np.sqrt(pnu**2 + self.mbetasq(m1, m2, m3, order))
        vnu = pnu / Enu
        sigma = self.sigmav(Ee) / vnu
        Ue1sq, Ue2sq, Ue3sq = self.Ue(order)
        return sigma * NT / np.pi**2 * (Ue1sq * self.H(Ee, m1, DEEnd=DEEnd) + Ue2sq * self.H(Ee, m2, DEEnd=DEEnd) + Ue3sq * self.H(Ee, m3, DEEnd=DEEnd))

    def CNBfactor(self, mi, mlightest, Ee, delta, DEEnd=0.0):
        Eend = self.Eend0() - mlightest + DEEnd
        return np.exp(-(Ee - (Eend + mi + mlightest))**2 / (2 * delta**2/(8 * np.log(2))))

    def dGtCNBdE(self, delta, m1, m2, m3, Ee, n0, NT, order='normal', DEEnd=0.0):
        '''
        Equation 3.10 - smeared CNB rate
        '''
        mlightest = min(m1, m2, m3)
        Eend = self.Eend0() - mlightest
        Gamma1, Gamma2, Gamma3 = self.GammaCNB(m1, m2, m3, Ee, n0, NT, order, total=False)
        E1, E2, E3 = self.CNBfactor(m1, mlightest, Ee, delta, DEEnd=DEEnd), self.CNBfactor(m2, mlightest, Ee, delta, DEEnd=DEEnd), self.CNBfactor(m3, mlightest, Ee, delta, DEEnd=DEEnd)
        return np.sqrt(8 * np.log(2)) / (np.sqrt(2 * np.pi) * delta) * (Gamma1 * E1 + Gamma2 * E2 + Gamma3 * E3)

    def dGtbdE(self, delta, m1, m2, m3, Ee, NT, order='normal', DEEnd=0.0):
        '''
        Equation 3.11 - smeared beta decay rate
        '''
        me = 511 * 1e6 # meV
        prefactor = np.sqrt(8 * np.log(2)) / (np.sqrt(2 * np.pi) * delta)
        f = lambda x: self.dGbdEe(m1, m2, m3, x, NT, order, DEEnd=DEEnd) * np.exp(-(Ee - x)**2/(2 * delta**2 / (8 * np.log(2))))
        integral = self.integrator(f=f, a=Ee - 10 * delta, b=Ee + 10 * delta)
        return prefactor * integral

    def N_beta(self, Ei, Tyrs, delta, mlightest, NT, order='normal', DEEnd=0.0):
        m1, m2, m3 = self.masses(mlightest, order)
        eVyr_factor = 4.794049023619834e+22
        f = lambda x: eVyr_factor * self.dGtbdE(delta, m1, m2, m3, x, NT, order, DEEnd=DEEnd)
        integral = self.integrator(f=f, a=Ei - 0.5 * delta, b=Ei + 0.5 * delta)
        return Tyrs * 1e-3 * integral

    def N_CNB(self, Ei, Tyrs, delta, mlightest, n0, NT, order='normal', DEEnd=0.0):
        m1, m2, m3 = self.masses(mlightest, order)
        eVyr_factor = 4.794049023619834e+22
        n0 = n0 * 7.685803257085992e-06
        f = lambda x: eVyr_factor * self.dGtCNBdE(delta, m1, m2, m3, x, n0, NT, order, DEEnd=DEEnd)
        integral = self.integrator(f=f, a=Ei - 0.5 * delta, b=Ei + 0.5 * delta)
        return Tyrs * 1e-3 * integral

    def N_total(self, Ei, Tyrs, delta, mlightest, n0, NT, order='normal', DEEnd=0.0, gamma_b=1e-5):
        return self.N_beta(Ei, Tyrs, delta, mlightest, NT, order, DEEnd=DEEnd) + self.N_CNB(Ei, Tyrs, delta, mlightest, n0, NT, order, DEEnd=DEEnd) + (gamma_b * 31558149.7635456) * Tyrs / (15 * 1e3 / 50)

    def masses(self, mlightest, order='normal'):
        if order == 'normal':
            m1 = mlightest
            m2 = np.sqrt(m1**2 + 7.55 * 1e-5 * 1e6)
            m3 = np.sqrt(m1**2 + 2.50 * 1e-3 * 1e6)
        else:
            m3 = mlightest
            m1 = np.sqrt(m3**2 + 2.42 * 1e-3 * 1e6)
            m2 = np.sqrt(m1**2 + 7.55 * 1e-5 * 1e6)
        return m1, m2, m3

    def loglkl(self, cosmo, data):

        A_beta = data.mcmc_parameters['A_beta']['current']
        A_CNB = data.mcmc_parameters['A_CNB']['current']
        Nb = np.exp(data.mcmc_parameters['lnNb']['current'])
        mlightest = data.mcmc_parameters['mlightest']['current']
        DEEnd = data.mcmc_parameters['DEEnd']['current']

        Nth_arr = Nb + A_beta * self.N_beta(self.Ei_arr, self.Tyrs, self.delta, mlightest, self.NT, self.order, DEEnd=DEEnd) + A_CNB * self.N_CNB(self.Ei_arr, self.Tyrs, self.delta, mlightest, self.n0, self.NT, self.order, DEEnd=DEEnd)

        # Poisson Likelihood
        return np.sum(self.Nt_arr * np.log(Nth_arr) - Nth_arr - self.ln_Nt_factorial)

        # Gaussian Likelihood
        # return -0.5 * np.sum(np.power((self.Nt_arr - Nth_arr)/np.sqrt(self.Nt_arr), 2.0))