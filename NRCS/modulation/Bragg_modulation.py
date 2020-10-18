import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import fv
from NRCS.model.Bragg_scattering import MSS
from NRCS.modulation import Spectrum
from NRCS.modulation.Spectrum import wn_exp, wind_exp, sample_back

def ni2_func(kr, k, K, u_10, fetch, azimuth, tsc, wind_dir):
    B_old, B_new = Spectrum(k, K, u_10, fetch, azimuth, tsc, wind_dir)
    k_re, wind = sample_back(k, u_10, tsc)
    kd = const.d*kr
    ni2 = np.zeros([k_re.shape[0],k_re.shape[1]])
    for ii in np.arange(k_re.shape[0]):
        for jj in np.arange(k_re.shape[1]):
            kk = k_re[ii, jj,:]
            ni2[ii, jj] = np.trapz(B_new[ii,jj,:][kk >= kd] / kk[kk >= kd], kk[kk >= kd])
    return ni2

def Trans_func(k, K, u_10, fetch, azimuth, divergence):
    nk = k.shape[0]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = k*fv(u_10)**2/const.g
    K_hat = K * fv(u_10) ** 2 / const.g
    wind_exponent = wind_exp(k)
    mk = wn_exp(k.reshape(nk, 1), u_10, fetch, azimuth)[:, 0]
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10) * divergence / (const.g*(
                1 + 1j * c_tau * k_hat ** (-2) * K_hat))
    return T

def CPBr_new(kr, K, theta, azimuth, wind, ind, fetch, divergence):
    # [Kudryavtsev, 2019] equation A1c
    nphi = theta.shape[0]
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta)**2)
    Gvv = np.cos(theta)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta)**2) - np.sin(theta)**2) / (const.epsilon_sw*np.cos(theta)+eps_sin)**2
    Ghh = np.cos(theta)**2*(const.epsilon_sw-1)/(np.cos(theta)+eps_sin)**2
    G = np.abs(Gvv-Ghh)**2
    kbr = 2*kr*np.sin(theta)
    sn2 = MSS(kbr, wind, fetch)
    Bkdir = np.zeros([wind.shape[0], wind.shape[1]])
    for ii in np.arange(wind.shape[0]):
        for jj in np.arange(wind.shape[1]):
            T = Trans_func(kbr[:, 0], K[ii, jj], wind[ii, jj], fetch, azimuth, divergence[ii, jj]).reshape(nphi,1)
            Bkdir[ii, jj] = (kudryavtsev05(kbr.reshape(nphi, 1), wind[ii, jj], fetch, azimuth)*(1+abs(T)))[jj, ind]
    Brcp = np.pi * G * sn2 * Bkdir/(np.tan(theta)**4 * np.sin(theta)**2)
    return Brcp

def Br_new(k, K, kr, theta, azimuth, u_10, fetch, wind_dir, ind_pi, tsc, polarization):
    """
    :param k:
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :return:
    """

    nk = k.shape[0]
    # divergence of the sea surface current
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)

    k_re, wind = sample_back(k, u_10, tsc)

    # wind direction index
    ind = np.where(np.degrees(azimuth) == wind_dir)[0]

    if polarization == 'VH':
        Br = CPBr_new(kr, K, theta, azimuth, wind, ind, fetch, divergence)
        return Br

    # # Sea surface slope in the direction of incidence angle
    ni2 = ni2_func(k, K, u_10, fetch, azimuth, tsc, wind_dir)

    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    ni = ni.reshape(nk, 1)
    ni = np.tan(ni)

    Br = np.zeros([tsc.shape[0], tsc.shape[1]])

    for ii in np.arange(ni2.shape[0]):
        for jj in np.arange(ni2.shape[1]):
            P = np.exp(-0.5 * (ni - np.mean(ni)) ** 2 / ni2[ii, jj]) / np.sqrt(2 * np.pi * ni2[ii, jj])
            #  the range of the sea surface slope
            angle_index = np.logical_and(-3 * 180 * np.arctan(np.sqrt(ni2[ii ,jj])) / np.pi < np.arctan(ni) * 180 / np.pi, np.arctan(ni) * 180 / np.pi < 3 * 180 * np.arctan(np.sqrt(ni2[ii, jj])) / np.pi)
            P = P[angle_index]
            nini = ni[angle_index]
            nnk = nini.shape[0]
            nini = nini.reshape(nnk, 1)
            # local incidence angle
            theta_l = np.abs(theta[jj] - np.arctan(nini).reshape(nnk, 1))
            kbr = 2*kr*np.sin(theta_l)
            # geometric scattering coefficients [Plant 1997] equation 5,6
            eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)
            kkbr = np.sort(kbr)
            T = Trans_func(kkbr[:,0], K[ii, jj], wind[ii, jj], fetch, azimuth, divergence[ii, jj])[np.argsort(kbr)]
            spec_Skk = kudryavtsev05(kkbr.reshape(nnk, 1), wind[ii,jj], fetch, azimuth)[np.argsort(kbr)[:,0], :] * (1+abs(T)) / kbr.reshape(nnk, 1) ** 4
            Skb_r = (spec_Skk[:, ind] + spec_Skk[:, ind_pi]) / 2
            if polarization == 'VV':
                G = np.cos(theta_l)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta_l)**2) - np.sin(theta_l)**2) / (const.epsilon_sw*np.cos(theta_l)+eps_sin)**2
                G = np.abs(G)**2
            else:
                G = np.cos(theta_l)**2*(const.epsilon_sw-1)/(np.cos(theta_l)+eps_sin)**2
                G = np.abs(G)**2
            # pure Bragg scattering NRCS
            br0 = 16 * np.pi * kr ** 4 * G * Skb_r
            # Bragg scattering composite model
            BR = br0 * P.reshape(nnk,1)
            # integral over kbr >= kd
            a = np.tan(theta[jj] - const.d / 2)
            b = np.tan(theta[jj] + const.d / 2)
            Br[ii, jj] = np.trapz(BR[nini <= a], nini[nini <= a]) + np.trapz(BR[nini >= b], nini[nini >= b])
    return Br
