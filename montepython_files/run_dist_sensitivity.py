import numpy as np
import subprocess
import glob

FD_file = 'nudist_forecast/massive_FD.param'
HEDR_file = 'nudist_forecast/highEnergy_plus_DR.param'
HELT_file = 'nudist_forecast/highEnergy_plus_lowT.param'
HE_file = 'nudist_forecast/highEnergy.param'
LEDR_file = 'nudist_forecast/lowEnergy_plus_DR.param'

def change_file(filename, mncdm):
    file = open(filename, "r")
    lines = file.readlines()
    lines[10] = "data.parameters['m_ncdm']\t= [{}, 0., 5., 0.1, 1., 'cosmo']\n".format(mncdm)
    file.close()
    file = open(filename, "w")
    file.writelines(lines)
    file.close()

def change_experiment(expt):
    for filename in [FD_file, HEDR_file, HELT_file, HE_file, LEDR_file]:
        file = open(filename, "r")
        lines = file.readlines()
        if expt == 'planck':
            lines[2] = "data.experiments=['fake_planck_bluebook']\n"
        elif expt == 'cmb-s4':
            lines[2] = "data.experiments=['cmb_s4']\n"
        elif expt == 'planck+cmb-s4':
            lines[2] = "data.experiments=['fake_planck_bluebook', 'cmb_s4']\n"
        file.close()
        file = open(filename, "w")
        file.writelines(lines)
        file.close()

def get_likelihood(folder):
    if folder != 'MFD':
        files = glob.glob(folder + '/*__1.txt')
    elif folder == 'MFD':
        files = glob.glob(folder + '/*__2.txt')
    data = np.loadtxt(files[0])
    try:
        likelihood = data[1]
        return likelihood
    except:
        return np.nan

def sigma(lik):
    if lik != np.nan:
        return np.sqrt(2 * max(0.0, lik))
    else:
        return np.nan

if __name__ == '__main__':
    expt = 'cmb-s4' # or 'planck'
    change_experiment(expt)
    mass_arr = np.linspace(0.0, 0.2, 100)    

    for idx, mncdm in enumerate(mass_arr):
        print("mncdm = {}".format(mncdm))
        change_file(FD_file, mncdm)
        change_file(HEDR_file, 10 * mncdm)
        change_file(HELT_file, mncdm)
        change_file(HE_file, 10 * mncdm)
        change_file(LEDR_file, mncdm)
        subprocess.call("sh run_dist_sensitivity.sh", shell=True, stdout=subprocess.PIPE)
        HEDR_lik = get_likelihood('HEDR')
        HELT_lik = get_likelihood('HELT')
        HE_lik = get_likelihood('HE')
        LTDR_lik = get_likelihood('LTDR')
        MFD_lik = get_likelihood('MFD')
        to_add = [mncdm, sigma(MFD_lik), sigma(HEDR_lik), sigma(HELT_lik), sigma(HE_lik), sigma(LTDR_lik)]
        if idx == 0:
            output_arr = np.array([to_add])
        else:
            output_arr = np.vstack([output_arr, to_add])
        np.savetxt('{}_sensitivity.txt'.format(expt), output_arr, header='mncdm MFD HEDR HELT HE LTDR')
        print(to_add)
