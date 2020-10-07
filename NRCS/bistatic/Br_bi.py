from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import numpy as np
from NRCS.model.Bragg_scattering import MSS
from NRCS.spec import kudryavtsev05
from scipy.interpolate import interp2d
from stereoid.oceans.bistatic_pol import elfouhaily

def B_int(k, u_10, fetch, azi):
    if azi.max() > np.pi:
        azimuth = np.linspace(0, 2 * np.pi, 19*2-1)# 1024
    elif azi.min() < -np.pi:
        azimuth = np.linspace(-2 * np.pi, 0, 19*2-1)
    else:
        azimuth = np.linspace(-np.pi, np.pi, 19 * 2 - 1)  # 1024
    B = kudryavtsev05(k, u_10, fetch, azimuth)
    f = interp2d(azimuth, k[:, 0], B, kind='cubic')
    return f(azi, k[:, 0])

def eq_CPBr(kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, spec_name):
    # [Kudryavtsev, 2019] equation A1c
    nphi = theta_eq.shape[0]
    theta_eq = theta_eq.reshape(nphi, 1)
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_eq)**2)
    Gvv = np.cos(theta_eq)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta_eq)**2) - np.sin(theta_eq)**2) / (const.epsilon_sw*np.cos(theta_eq)+eps_sin)**2
    Ghh = np.cos(theta_eq)**2*(const.epsilon_sw-1)/(np.cos(theta_eq)+eps_sin)**2
    G = (np.abs(Gvv-Ghh)**2).reshape(nphi, 1)
    kbr = 2*kr*np.sin(theta_eq) * np.cos((bist_ang_az/2).reshape(nphi, 1))
    sn2 = MSS(kbr, u_10, fetch).reshape(nphi,1)
    specf = spec.models[spec_name]
    ind = np.arange(nphi)

    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        Bkdir = specf(kbr.reshape(nphi, 1), u_10, fetch) * spreadf(kbr.reshape(nphi, 1), eq_azi, u_10, fetch) * kbr.reshape(nphi, 1)**3
    else:
        sort_ind = np.argsort(eq_azi)
        sort_azi = np.sort(eq_azi)
        # BB = specf(kbr.reshape(nphi, 1), u_10, fetch, eq_azi)
        BB = B_int(kbr.reshape(nphi, 1), u_10, fetch, sort_azi)[:, sort_ind]
        Bkdir = BB[ind, ind]
    Brcp = np.pi * G * sn2 * Bkdir.reshape(nphi,1)/(np.tan(theta_eq)**4 * np.sin(theta_eq)**2)
    return Brcp[:, 0]

def eq_br(k, kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, spec_name, polarization):
    """
    all the angles are in radians
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
        Br = eq_CPBr(kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, spec_name)
        return Br

    nphi = theta_eq.shape[0]
    nk = k.shape[0]
    nazi = eq_azi.shape[0]
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
    theta_l = np.abs(theta_eq - np.arctan(ni).reshape(nnk, 1))

    # geometric scattering coefficients [Plant 1997] equation 5,6
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)

    if polarization == 'VV':
        G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) * (
                    const.epsilon_sw * (1 + np.sin(theta_l) ** 2) - np.sin(theta_l) ** 2) / (
                        const.epsilon_sw * np.cos(theta_l) + eps_sin) ** 2
        G = np.abs(G) ** 2
    else:
        G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) / (np.cos(theta_l) + eps_sin) ** 2
        G = np.abs(G) ** 2

    # compute the wave number for bistatic geometry
    kbr = 2 * kr * np.sin(theta_l) * np.cos(bist_ang_az/2)
    # sort eq_azi
    sort_ind = np.argsort(eq_azi)
    sort_azi = np.sort(eq_azi)
    if spec_name == 'kudryavtsev05':
        Skb = np.zeros([nnk, nphi])
        Skb_pi = np.zeros([nnk, nphi])
        for nn in np.arange(nazi):
            # spec_Skk1 = specf(kbr[:, nn].reshape(nnk, 1), u_10, fetch, sort_azi) / kbr[:, nn].reshape(nnk, 1) ** 4
            spec_Skk1 = B_int(kbr[:, nn].reshape(nnk, 1), u_10, fetch, sort_azi)[:, sort_ind] / kbr[:, nn].reshape(nnk, 1) ** 4
            Skb[:, nn] = spec_Skk1[:, nn]
            # spec_Skk2 = specf(kbr[:, nn].reshape(nnk, 1), u_10, fetch, np.pi + sort_azi) / kbr[:, nn].reshape(nnk, 1) ** 4
            spec_Skk2 = B_int(kbr[:, nn].reshape(nnk, 1), u_10, fetch, np.pi + sort_azi)[:, sort_ind] / kbr[:, nn].reshape(nnk, 1) ** 4
            Skb_pi[:, nn] = spec_Skk2[:, nn]
    else:
        Sk = specf(kbr, u_10, fetch)
        spreadf = spread.models[spec_name]
        Skb = Sk * spreadf(kbr, eq_azi, u_10, fetch) / kbr  # equation 45
        Skb_pi = Sk * spreadf(kbr, eq_azi + np.pi, u_10, fetch) / kbr  # equation 45
    Skb_r = (Skb + Skb_pi) / 2  # Kudryavtsev 2003a equation 2

    # pure Bragg scattering NRCS
    br0 = 16 * np.pi * kr ** 4 * G * Skb_r

    # Bragg scattering composite model
    BR = br0 * P.reshape(nnk, 1)

    # integral over kbr >= kd
    intebr = []

    for i in np.arange(nphi):
        a = np.tan(theta_eq[i] - const.d / (2 * np.cos(bist_ang_az[i])))
        b = np.tan(theta_eq[i] + const.d / (2 * np.cos(bist_ang_az[i])))
        inte = BR[:, i].reshape(nnk, 1)
        vv = np.trapz(inte[ni <= a], ni[ni <= a]) + np.trapz(inte[ni >= b], ni[ni >= b])
        intebr.append(vv)
    Br = np.asarray(intebr)
    return Br

def pol_vec_br(theta_i, theta_s, bist_ang_az, eq_azi, inc_polar):
    """
    :param theta_i:
    :param eq_azi:
    :param polarization:
    :return:
    """
    # polarization of incident plane
    if inc_polar == 'V':
        poli = 90
    elif inc_polar == 'H':
        poli = 0
    # polarization for the mono-static equivalent
    (rot_ang_1, rot_ang_2, rot_ang_tot, Ps1, Ps2, Ps_tot) = elfouhaily(poli, np.degrees(eq_azi), np.degrees(theta_i), np.degrees(eq_azi), np.degrees(theta_i))
    Ps_eq_norm = np.linalg.norm(Ps_tot, axis=-1)
    (rot_ang_1, rot_ang_2, rot_ang_tot, Ps1, Ps2, Ps_tot) = elfouhaily(poli, np.degrees(eq_azi-bist_ang_az/2), np.degrees(theta_i), np.degrees(eq_azi+bist_ang_az/2), np.degrees(theta_s))
    Ps_bi_norm = np.linalg.norm(Ps_tot, axis=-1)
    #  transfer function
    M = Ps_bi_norm ** 2 / Ps_eq_norm ** 2
    return M

def Br_bi(k, kr, theta_i, theta_s, theta_eq, bist_ang_az, eq_azi, u_10, fetch, spec_name, polarization, inc_polar):
    M = pol_vec_br(theta_i, theta_s, bist_ang_az, eq_azi, inc_polar)
    Br = eq_br(k, kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, spec_name, polarization)
    return M * Br
