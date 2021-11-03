from math import nan
import numpy as np
from scipy.special import gamma
import functools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(__call__, obj)

# Define order of Gaussian quadrature integration
points, weights = np.polynomial.legendre.leggauss(50)

@vectorize
def integrator(f, a, b):
    sub = (b - a) / 2.
    add = (b + a) / 2.
    if sub == 0:
        return 0.
    return sub * np.dot(f(sub * points + add), weights)

def F(Ee):
    '''
    Equation 3.6 - Fermi function F(Z, Ee) with Z = 2
    '''
    me = 511 * 1e6 # meV
    pe = np.sqrt(Ee**2 - me**2)
    Z = 2
    alpha = 1 / 137.036
    eta = Z * alpha * Ee / pe
    return 2 * np.pi * eta / (1 - np.exp(- 2 * np.pi * eta))

def sigmav(Ee):
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
    return (GF**2 / (2 * np.pi)) * F(Ee) * (m3He/m3H) * Ee * pe * (F2 + gA**2 * GT2)

def Ue(order='normal'):
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

def fc(mi_meV, dist="Gaussian"):
    '''
    Equation 2.3 - gravitational clustering factors
    '''
    if dist == "Gaussian":
        return 0.
    return 76.5 * (mi_meV / 1000)**(2.21)

def GammaCNB(m1, m2, m3, Ee, n0, NT, order='normal', total=True):
    Ue1sq, Ue2sq, Ue3sq = Ue(order)
    Gamma1 = NT * sigmav(Ee) * n0 * Ue1sq * (1 + fc(m1))
    Gamma2 = NT * sigmav(Ee) * n0 * Ue2sq * (1 + fc(m2))
    Gamma3 = NT * sigmav(Ee) * n0 * Ue3sq * (1 + fc(m3))
    if total:
        return Gamma1 + Gamma2 + Gamma3
    else:
        return Gamma1, Gamma2, Gamma3

def mbetasq(m1, m2, m3, order='normal'):
    '''
    Equation 2.1 - effective neutrino beta decay mass
    '''
    Ue1sq, Ue2sq, Ue3sq = Ue(order)
    return Ue1sq * m1**2 + Ue2sq * m2**2 + Ue3sq * m3**2

def Eend0():
    '''
    From 0706.0897 (eq. 2)
    '''
    m3He = 2808.391 * 1e9 # meV
    m3H = 2808.921 * 1e9 # meV
    me = 511 * 1e6 # meV
    return (1/(2 * m3H)) * (m3H**2 + me**2 - m3He**2)

def H(Ee, mi, DEEnd=0.0):
    '''
    Equation 3.9 - H(Ee, mi) function
    '''
    y = np.heaviside(Eend0() + DEEnd - Ee - mi, 0) * (Eend0() + DEEnd- Ee - mi)
    m3He = 2808.391 * 1e9 # meV
    m3H = 2808.921 * 1e9 # meV
    me = 511 * 1e6 # meV
    return (1 - me**2/(Ee * m3H)) * np.sqrt(y * (y + (2 * mi * m3He)/(m3H))) * np.power(1 - 2 * Ee / m3H + me**2 / m3H**2, -2.0) * (y + (mi/m3H)*(m3He + mi))

def dGbdEe(m1, m2, m3, Ee, NT, order='normal', DEEnd=0.0):
    '''
    Equation 3.8 - Beta decay rate
    '''
    me = 511 * 1e6 # meV
    pe = np.sqrt(Ee**2 - me**2)
    pnu = pe
    Enu = np.sqrt(pnu**2 + mbetasq(m1, m2, m3, order))
    vnu = pnu / Enu
    sigma = sigmav(Ee) / vnu
    Ue1sq, Ue2sq, Ue3sq = Ue(order)
    return sigma * NT / np.pi**2 * (Ue1sq * H(Ee, m1, DEEnd=DEEnd) + Ue2sq * H(Ee, m2, DEEnd=DEEnd) + Ue3sq * H(Ee, m3, DEEnd=DEEnd))

def CNBfactor(mi, mlightest, Ee, delta, DEEnd=0.0):
    Eend = Eend0() - mlightest + DEEnd
    return np.exp(-(Ee - (Eend + mi + mlightest))**2 / (2 * delta**2/(8 * np.log(2))))

def dGtCNBdE(delta, m1, m2, m3, Ee, n0, NT, order='normal', DEEnd=0.0):
    '''
    Equation 3.10 - smeared CNB rate
    '''
    mlightest = min(m1, m2, m3)
    Eend = Eend0() - mlightest
    Gamma1, Gamma2, Gamma3 = GammaCNB(m1, m2, m3, Ee, n0, NT, order, total=False)
    E1, E2, E3 = CNBfactor(m1, mlightest, Ee, delta, DEEnd=DEEnd), CNBfactor(m2, mlightest, Ee, delta, DEEnd=DEEnd), CNBfactor(m3, mlightest, Ee, delta, DEEnd=DEEnd)
    return np.sqrt(8 * np.log(2)) / (np.sqrt(2 * np.pi) * delta) * (Gamma1 * E1 + Gamma2 * E2 + Gamma3 * E3)

def dGtbdE(delta, m1, m2, m3, Ee, NT, order='normal', DEEnd=0.0):
    '''
    Equation 3.11 - smeared beta decay rate
    '''
    me = 511 * 1e6 # meV
    prefactor = np.sqrt(8 * np.log(2)) / (np.sqrt(2 * np.pi) * delta)
    f = lambda x: dGbdEe(m1, m2, m3, x, NT, order, DEEnd=DEEnd) * np.exp(-(Ee - x)**2/(2 * delta**2 / (8 * np.log(2))))
    integral = integrator(f=f, a=Ee - 10 * delta, b=Ee + 10 * delta)
    return prefactor * integral

def N_beta(Ei, Tyrs, delta, mlightest, NT, order='normal', DEEnd=0.0):
    m1, m2, m3 = masses(mlightest, order)
    eVyr_factor = 4.794049023619834e+22
    f = lambda x: eVyr_factor * dGtbdE(delta, m1, m2, m3, x, NT, order, DEEnd=DEEnd)
    integral = integrator(f=f, a=Ei - 0.5 * delta, b=Ei + 0.5 * delta)
    return Tyrs * 1e-3 * integral

def N_CNB(Ei, Tyrs, delta, mlightest, n0, NT, order='normal', DEEnd=0.0, cDM=1.):
    m1, m2, m3 = masses(mlightest, order)
    eVyr_factor = 4.794049023619834e+22
    n0 = n0 * 7.685803257085992e-06
    f = lambda x: eVyr_factor * dGtCNBdE(delta, m1, m2, m3, x, n0, NT, order, DEEnd=DEEnd)
    integral = integrator(f=f, a=Ei - 0.5 * delta, b=Ei + 0.5 * delta)
    return cDM * Tyrs * 1e-3 * integral

def N_total(Ei, Tyrs, delta, mlightest, n0, NT, order='normal', DEEnd=0.0, gamma_b=1e-5, cDM=1.):
    return N_beta(Ei, Tyrs, delta, mlightest, NT, order, DEEnd=DEEnd) + N_CNB(Ei, Tyrs, delta, mlightest, n0, NT, order, DEEnd=DEEnd, cDM=cDM) + (gamma_b * 31558149.7635456) * Tyrs / (15 * 1e3 / 50)

def masses(mlightest, order='normal'):
    if order == 'normal':
        m1 = mlightest
        m2 = np.sqrt(m1**2 + 7.55 * 1e-5 * 1e6)
        m3 = np.sqrt(m1**2 + 2.50 * 1e-3 * 1e6)
    else:
        m3 = mlightest
        m1 = np.sqrt(m3**2 + 2.42 * 1e-3 * 1e6)
        m2 = np.sqrt(m1**2 + 7.55 * 1e-5 * 1e6)
    return m1, m2, m3


if __name__ == '__main__':
    # User Input
    Tyrs = float(input("T [yrs] (default 1 yr): "))
    delta = float(input("Delta [meV] (default 100 meV): "))
    order = str(input("Mass ordering (default normal) : "))
    mT = float(input("Tritium mass [g] (default 100g): "))
    NT = 1.9972819100287977e+25 * (mT / 100.)
    gamma_b = float(input("Gamma_b [Hz] (default 1e-5 Hz): "))
    spin = str(input("Spin (default Dirac): "))
    if spin.lower()[0] == 'd':
        cDM_sim = 1.0
    else:
        cDM_sim = 2.0
    border = 20
    Elow, Ehigh = -5000.0, 10000.0
    Ei_arr = np.linspace(Eend0() + Elow, Eend0() + Ehigh, int((Ehigh - Elow)/delta))
    Nb_data = 1.05189 * (gamma_b/1e-5) * (Tyrs / 1.0)

    filename = '[t]{}_[d]{}_[m]{}_[o]{}_[s]{}_[b]{}.txt'.format(Tyrs, delta, mT, order.lower()[0], spin.lower()[0], gamma_b)

    print('Results will be saved to:', filename, '\n')

    mlight_min = float(input("min(mlightest) [meV] (default 10 meV): "))
    mlight_max = float(input("max(mlightest) [meV] (default 1000 meV): "))
    mlight_num = int(input("num(mlightest) (default 50): "))
    nloc_min = float(input("min(nloc) [cm-3] (default 1 cm-3): "))
    nloc_max = float(input("max(nloc) [cm-3] (default 1000 cm-3): "))
    nloc_num = int(input("num(nloc) (default 50): "))

    mlight, nloc = np.meshgrid(np.geomspace(mlight_min, mlight_max, mlight_num), np.geomspace(nloc_min, nloc_max, nloc_num))
    mlight = mlight.reshape(1, -1)[0]
    nloc = nloc.reshape(1, -1)[0]


    def denominator(nloc, mlight, Nb, A_beta, DEEnd, Ndata_arr, ln_Ndata_factorial):
        A_CNB = 1.
        Nth_arr = Nb + A_beta * N_beta(Ei_arr, Tyrs, delta, mlight, NT, order, DEEnd=DEEnd) + A_CNB * N_CNB(Ei_arr, Tyrs, delta, mlight, nloc, NT, order, DEEnd=DEEnd, cDM=cDM_sim)
        ll1 = np.sum(Ndata_arr * np.log(Nth_arr) - Nth_arr - ln_Ndata_factorial)
        return -ll1

    def numerator(params, Ndata_arr, ln_Ndata_factorial):
        mass, Nb, A_beta, DEEnd = params

        Nth_arr = Nb + A_beta * N_beta(Ei_arr, Tyrs, delta, mass, NT, order, DEEnd=DEEnd)
        ll0 = np.sum(Ndata_arr * np.log(Nth_arr) - Nth_arr - ln_Ndata_factorial)
        return -ll0

    def func(nloc, mlight):

        Ndata_arr = N_total(Ei_arr, Tyrs, delta, mlight, nloc, NT, order=order, DEEnd=0.0, gamma_b=gamma_b, cDM=cDM_sim)
        ln_Ndata_factorial = np.zeros(len(Ndata_arr))
        ln_Ndata_factorial[Ndata_arr < border] = np.log(gamma(Ndata_arr[Ndata_arr < border] + 1))
        ln_Ndata_factorial[Ndata_arr > border] = Ndata_arr[Ndata_arr > border] * np.log(Ndata_arr[Ndata_arr > border]) - Ndata_arr[Ndata_arr > border] + np.log(1./30 + Ndata_arr[Ndata_arr > border] * (1 + 4 * Ndata_arr[Ndata_arr > border] * (1 + 2 * Ndata_arr[Ndata_arr > border]))) / 6. + 0.5 * np.log(np.pi)

        loglkl_denominator = -1 * denominator(nloc, mlight, Nb_data, 1., 0., Ndata_arr, ln_Ndata_factorial)

        initial_guess = [mlight, Nb_data, 1., 1e-5]
        numerator_opt = optimize.minimize(numerator, initial_guess, args=(Ndata_arr, ln_Ndata_factorial), method='Powell', options={'xtol': 1e-3, 'ftol': 1e-3})
        if numerator_opt.success:
            bestfit_params_numerator = numerator_opt.x
            loglkl_numerator = -numerator(bestfit_params_numerator, Ndata_arr, ln_Ndata_factorial)
        else:
            print("Numerator minimisation failed")
            loglkl_numerator = nan
            bestfit_params_numerator = [0., 0., 0., 0.]

        lambdaasq = -2 * loglkl_numerator + 2 * loglkl_denominator

        print(nloc, mlight, lambdaasq, np.sqrt(max(0.0, lambdaasq)), bestfit_params_numerator)

        return np.sqrt(max(0.0, lambdaasq))

    result = Parallel(n_jobs=-1)(delayed(func)(n, m) for n, m in zip(nloc, mlight))
    to_save = np.vstack([mlight, nloc, result]).T
    np.savetxt(filename, to_save, header='mlight/meV nloc/cm-3 sensitivity')
