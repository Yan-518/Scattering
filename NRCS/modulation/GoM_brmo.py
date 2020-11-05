import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.modulation.Bragg_modulation import CPBr_new, Trans_func
from NRCS.modulation import GoM_spec

def GoM_ni2_func(kr, k, K, u_10, fetch, azimuth, div, wind_dir):
    B_old, B_new = GoM_spec(k, K, u_10, fetch, azimuth, div, wind_dir)
    kd = const.d*kr
    ni2 = np.zeros([u_10.shape[0],u_10.shape[1]])
    for ii in np.arange(u_10.shape[0]):
        for jj in np.arange(u_10.shape[1]):
            kk = k[ii, jj,:]
            ni2[ii, jj] = np.trapz(B_new[ii,jj,:][kk >= kd] / kk[kk >= kd], kk[kk >= kd])
    return ni2

def GoM_brmo(k, K, kr, theta, azimuth, u_10, fetch, wind_dir, ind_pi, div, polarization):
    """
    :param k:
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :return:
    """

    nk = k.shape[2]

    # wind direction index
    ind = np.where(np.degrees(azimuth) == wind_dir)[0]

    if polarization == 'VH':
        Br = CPBr_new(kr, K, theta, azimuth, u_10, ind, fetch, div)
        return Br

    # # Sea surface slope in the direction of incidence angle
    ni2 = GoM_ni2_func(kr, k, K, u_10, fetch, azimuth, div, wind_dir)

    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    ni = ni.reshape(nk, 1)
    ni = np.tan(ni)

    Br = np.zeros([div.shape[0], div.shape[1]])

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
            T = Trans_func(kkbr[:, 0], K[ii, jj], u_10[ii, jj], fetch, azimuth, div[ii, jj])[np.argsort(kbr[:,0)].reshape(nnk,1)
            spec_Skk = kudryavtsev05(kkbr.reshape(nnk, 1), u_10[ii,jj], fetch, azimuth)[np.argsort(kbr[:,0]), :] * (1+abs(T)) / kbr.reshape(nnk, 1) ** 4
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
