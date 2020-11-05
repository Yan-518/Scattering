from NRCS import constants as const
import numpy as np
from NRCS.spec import kudryavtsev05
from scipy.interpolate import interp2d
from NRCS.bistatic.Br_bi import pol_vec_br
from NRCS.modulation.Bragg_modulation import Trans_func
from NRCS.modulation.GoM_brmo import GoM_ni2_func


def B_int_single(k, u_10, fetch, azi):
    if azi > np.pi:
        azimuth = np.linspace(0, 2 * np.pi, 721)# 1024
    elif azi < -np.pi:
        azimuth = np.linspace(-2 * np.pi, 0, 721)
    else:
        azimuth = np.linspace(-np.pi, np.pi, 721)  # 1024
    B = kudryavtsev05(k, u_10, fetch, azimuth)
    f = interp2d(azimuth, k[:, 0], B, kind='cubic')
    return f(azi, k[:, 0])

def eq_br_mo(k, K, kr, theta_eq, bist_ang_az, eq_azi, wind_dir, u_10, fetch, div, polarization):
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

    nk = k.shape[2]

    # Set azimuth to compute transfer function
    azimuth = np.linspace(-np.pi, np.pi, 37)

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
            angle_index = np.logical_and(-3 * 180 * np.arctan(np.sqrt(ni2[ii,jj])) / np.pi < np.arctan(ni) * 180 / np.pi, np.arctan(ni) * 180 / np.pi < 3 * 180 * np.arctan(np.sqrt(ni2[ii,jj])) / np.pi)
            P = P[angle_index]
            nini = ni[angle_index]
            nnk = nini.shape[0]
            nini = nini.reshape(nnk, 1)
            # local incidence angle
            theta_l = np.abs(theta_eq[jj] - np.arctan(nini).reshape(nnk, 1))
            # geometric scattering coefficients [Plant 1997] equation 5,6
            eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)
            # compute the wave number for bistatic geometry
            kbr = 2 * kr * np.sin(theta_l) * np.cos(bist_ang_az[jj]/2)
            kkbr = np.sort(kbr)
            T = Trans_func(kkbr[:, 0], K[ii, jj], u_10[ii, jj], fetch, azimuth, div[ii, jj])[np.argsort(kbr[:,0])].reshape(nnk,1)
            Skb = B_int_single(kkbr.reshape(nnk, 1), u_10[ii, jj], fetch, eq_azi[jj])[np.argsort(kbr[:,0]), :] * (1+abs(T)) / kbr.reshape(nnk,1) ** 4
            Skb_pi = B_int_single(kkbr.reshape(nnk, 1), u_10[ii, jj], fetch, np.pi + eq_azi[jj])[np.argsort(kbr[:,0]), :] * (1+abs(T)) / kbr.reshape(nnk, 1) ** 4
            Skb_r = (Skb + Skb_pi) / 2  # Kudryavtsev 2003a equation 2
            if polarization == 'VV':
                G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) * (
                    const.epsilon_sw * (1 + np.sin(theta_l) ** 2) - np.sin(theta_l) ** 2) / (
                        const.epsilon_sw * np.cos(theta_l) + eps_sin) ** 2
                G = np.abs(G) ** 2
            else:
                G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) / (np.cos(theta_l) + eps_sin) ** 2
                G = np.abs(G) ** 2
            # pure Bragg scattering NRCS
            br0 = 16 * np.pi * kr ** 4 * G * Skb_r
            # Bragg scattering composite model
            BR = br0 * P.reshape(nnk, 1)
            # integral over kbr >= kd
            a = np.tan(theta_eq[jj] - const.d / (2 * np.cos(bist_ang_az[jj]/2)))
            b = np.tan(theta_eq[jj] + const.d / (2 * np.cos(bist_ang_az[jj]/2)))
            Br[ii, jj] = np.trapz(BR[nini <= a], nini[nini <= a]) + np.trapz(BR[nini >= b], nini[nini >= b])
    return Br

def Br_bi_modulation(k, K, kr, theta_i, theta_s, theta_eq, bist_ang_az, eq_azi, u_10, wind_dir, fetch, div, polarization, inc_polar, sat):
    M = pol_vec_br(theta_i, theta_s, bist_ang_az, eq_azi, inc_polar, sat)
    Br = eq_br_mo(k, K, kr, theta_eq, bist_ang_az, eq_azi, wind_dir, u_10, fetch, div, polarization)
    return M * Br
