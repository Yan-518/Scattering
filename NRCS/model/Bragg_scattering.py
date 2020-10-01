from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak

def MSS(kbr, u_10, fetch):
    # Kudryavtsev, 2019a
    alpha = u_10*np.sqrt(spec_peak(u_10, fetch)/const.g)
    kd = kbr / 4
    return 4.5*10**(-3)*np.log(alpha**(-2)*kd*u_10**2/const.g)/2

def CPBragg(kr, theta, azimuth, u_10, fetch, spec_name):
    # [Kudryavtsev, 2019] equation A1c
    nphi = theta.shape[0]
    theta = theta.reshape(nphi, 1)
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta)**2)
    Gvv = np.cos(theta)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta)**2) - np.sin(theta)**2) / (const.epsilon_sw*np.cos(theta)+eps_sin)**2
    Ghh = np.cos(theta)**2*(const.epsilon_sw-1)/(np.cos(theta)+eps_sin)**2
    G = (np.abs(Gvv-Ghh)**2).reshape(nphi, 1)
    kbr = 2*kr*np.sin(theta)
    sn2 = MSS(kbr, u_10, fetch).reshape(nphi,1)
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        Bkdir = specf(kbr.reshape(nphi, 1), u_10, fetch) * spreadf(kbr.reshape(nphi, 1), azimuth, u_10, fetch) * kbr.reshape(nphi, 1)**3
    else:
        Bkdir = specf(kbr.reshape(nphi, 1), u_10, fetch, azimuth)
    Brcp = np.pi * G * sn2 * Bkdir/(np.tan(theta)**4 * np.sin(theta)**2)
    return Brcp

def Bragg_scattering(k, kr, theta, azimuth, u_10, fetch, spec_name, polarization):
    """
    :param k:
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :param spec_name:
    :return:
    """
    if polarization == 'VH':
        Br = CPBragg(kr, theta, azimuth, u_10, fetch, spec_name)
        return Br.T

    nphi = theta.shape[0]
    nk = k.shape[0]
    nazi = azimuth.shape[0]
    kd = const.d*kr

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]


    # # Sea surface slope in the direction of incidence angle
    phi_inc = np.linspace(-np.pi, np.pi, nazi*2)# in radians wave direction relative to the incidence plane
    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        Skdir_ni = specf(k.reshape(nk, 1), u_10, fetch) * spreadf(k.reshape(nk, 1), phi_inc, u_10, fetch)/k.reshape(nk, 1) # equation 45
    else:
        Skdir_ni = specf(k.reshape(nk, 1), u_10, fetch, phi_inc) / k.reshape(nk, 1) ** 4
    ni = np.trapz(k.reshape(nk, 1)**3*np.cos(phi_inc)**2*Skdir_ni, phi_inc, axis=1)
    ni2 = np.trapz(ni[k >= kd], k[k >= kd])

    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    ni = ni.reshape(nk, 1)
    ni = np.tan(ni)
    P = np.exp(-0.5 * (ni - np.mean(ni)) ** 2 / ni2) / np.sqrt(2 * np.pi * ni2)
    #  the range of the sea surface slope
    angle_index = np.logical_and(-3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi < np.arctan(ni) * 180 / np.pi, np.arctan(ni) * 180 / np.pi < 3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi)
    P = P[angle_index]
    ni = ni[angle_index]

    nnk = ni.shape[0]
    ni = ni.reshape(nnk, 1)

    # local incidence angle
    theta_l = np.abs(theta - np.arctan(ni).reshape(nnk, 1))
    kbr = 2*kr*np.sin(theta_l)

    # geometric scattering coefficients [Plant 1997] equation 5,6
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)

    # 3-D Sk computed from kudryavtsev05
    if spec_name == 'kudryavtsev05':
        Skk = np.zeros([nnk, nphi, nazi])
        # spec_Skk = specf(np.sort(kbr[0, :]).reshape(nphi, 1), u_10, fetch, azimuth) / np.sort(kbr[0, :]).reshape(nphi, 1) ** 4
        for nn in np.arange(nnk):
            spec_Skk = specf(kbr[nn, :].reshape(nphi, 1), u_10, fetch, azimuth) / kbr[nn, :].reshape(nphi, 1) ** 4
            Skk[nn, :, :] = spec_Skk
            # spec_Skk = specf(kbr[nn, :].reshape(nphi, 1), u_10, fetch, azimuth+np.pi) / kbr[nn, :].reshape(nphi, 1) ** 4
            # Skk_pi[nn, :, :] = spec_Skk
        inc = np.where(azimuth >= 0)[0]
        incc = np.linspace(1, inc[0], inc[0])
        inc = np.hstack((inc, incc))
        Skk_pi = Skk[:, :, inc.astype(int)]

    if polarization == 'VV':
        G = np.cos(theta_l)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta_l)**2) - np.sin(theta_l)**2) / (const.epsilon_sw*np.cos(theta_l)+eps_sin)**2
        G = np.abs(G)**2
    else:
        G = np.cos(theta_l)**2*(const.epsilon_sw-1)/(np.cos(theta_l)+eps_sin)**2
        G = np.abs(G)**2

    Br = np.zeros([nazi, nphi])
    for num in np.arange(nazi):
         #  wave number folded folded spcetrum of the surface elevations
        if spec_name == 'elfouhaily':
            Sk = specf(kbr, u_10, fetch)
            spreadf = spread.models[spec_name]
            Skb = Sk * spreadf(kbr, azimuth[num], u_10, fetch) / kbr  # equation 45
            Skb_pi = Sk * spreadf(kbr, azimuth[num] + np.pi, u_10, fetch) / kbr  # equation 45
        else:
            Skb = Skk[:, :, num]  # equation 45
            Skb_pi = Skk_pi[:, :, num]
        Skb_r = (Skb+Skb_pi) / 2 # Kudryavtsev 2003a equation 2

    # pure Bragg scattering NRCS
        br0 = 16 * np.pi * kr ** 4 * G * Skb_r

    # Bragg scattering composite model
        BR = br0 * P.reshape(nnk, 1)

    # integral over kbr >= kd
        intebr = []

        for i in np.arange(nphi):
            a = np.tan(theta[i]-const.d / 2)
            b = np.tan(theta[i]+const.d / 2)
            inte = BR[:, i].reshape(nnk, 1)
            vv = np.trapz(inte[ni <= a], ni[ni <= a]) + np.trapz(inte[ni >= b], ni[ni >= b])
            intebr.append(vv)
        intebr = np.asarray(intebr)
        Br[num, :] = intebr
    return Br
