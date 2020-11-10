from NRCS import constants as const
import numpy as np
from NRCS.spec import kudryavtsev05
from scipy.interpolate import interp2d
from NRCS.bistatic.Br_bi import pol_vec_br


def B_int_bf(k, u_10, fetch, azi):
    if azi > np.pi:
        azimuth = np.linspace(0, 2 * np.pi, 721)  # 1024
    elif azi < -np.pi:
        azimuth = np.linspace(-2 * np.pi, 0, 721)
    else:
        azimuth = np.linspace(-np.pi, np.pi, 721)  # 1024
    B = kudryavtsev05(k, u_10, fetch, azimuth)
    f = interp2d(azimuth, k[:, 0], B, kind='linear')
    return f(azi, k[:, 0])


def eq_br_cali(k, kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, polarization):
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

    nk = k.shape[0]
    kd = const.d * kr

    # Spectral model

    # # Sea surface slope in the direction of incidence angle
    phi_inc = np.linspace(-np.pi, np.pi, 37 * 2)  # in radians wave direction relative to the incidence plane
    Skdir_ni = kudryavtsev05(k.reshape(nk, 1), u_10, fetch, phi_inc) / k.reshape(nk, 1) ** 4
    ni = np.trapz(k.reshape(nk, 1) ** 3 * np.cos(phi_inc) ** 2 * Skdir_ni, phi_inc, axis=1)
    ni2 = np.trapz(ni[k >= kd], k[k >= kd])

    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    ni = ni.reshape(nk, 1)
    ni = np.tan(ni)
    P = np.exp(-0.5 * (ni - np.mean(ni)) ** 2 / ni2) / np.sqrt(2 * np.pi * ni2)
    #  the range of the sea surface slope
    angle_index = np.logical_and(-3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi < np.arctan(ni) * 180 / np.pi,
                                 np.arctan(ni) * 180 / np.pi < 3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi)
    P = P[angle_index]
    ni = ni[angle_index]

    nnk = ni.shape[0]
    ni = ni.reshape(nnk, 1)

    # local incidence angle
    theta_l = np.abs(theta_eq - np.arctan(ni).reshape(nnk, 1))

    # geometric scattering coefficients [Plant 1997] equation 5,6
    eps_sin = np.sqrt(const.epsilon_sw - np.sin(theta_l) ** 2)

    if polarization == 'VV':
        G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) * (
                const.epsilon_sw * (1 + np.sin(theta_l) ** 2) - np.sin(theta_l) ** 2) / (
                    const.epsilon_sw * np.cos(theta_l) + eps_sin) ** 2
        G = np.abs(G) ** 2
    else:
        G = np.cos(theta_l) ** 2 * (const.epsilon_sw - 1) / (np.cos(theta_l) + eps_sin) ** 2
        G = np.abs(G) ** 2

    # compute the wave number for bistatic geometry
    kbr = 2 * kr * np.sin(theta_l) * np.cos(bist_ang_az / 2)
    # sort eq_azi
    kkbr = np.sort(kbr[:, 0])
    spec_Skk1 = B_int_bf(kkbr.reshape(nnk, 1), u_10, fetch, eq_azi)[np.argsort(kbr[:, 0]), :] / kbr.reshape(nnk, 1) ** 4
    spec_Skk2 = B_int_bf(kkbr.reshape(nnk, 1), u_10, fetch, np.pi + eq_azi)[np.argsort(kbr[:, 0]), :] / kbr.reshape(nnk, 1) ** 4
    Skb_r = (spec_Skk1 + spec_Skk2) / 2

    # pure Bragg scattering NRCS
    br0 = 16 * np.pi * kr ** 4 * G * Skb_r

    # Bragg scattering composite model
    BR = br0 * P.reshape(nnk, 1)

    # integral over kbr >= kd

    a = np.tan(theta_eq - const.d / (2 * np.cos(bist_ang_az / 2)))
    b = np.tan(theta_eq + const.d / (2 * np.cos(bist_ang_az / 2)))
    Br = np.trapz(BR[ni <= a], ni[ni <= a]) + np.trapz(BR[ni >= b], ni[ni >= b])
    return Br

def California_bi_bf_br(k, kr, theta_i, theta_s, theta_eq, bist_ang_az, eq_azi, u_10, fetch, polarization, inc_polar, sat):
    M = pol_vec_br(theta_i, theta_s, bist_ang_az, eq_azi, inc_polar, sat)
    Br = eq_br_cali(k, kr, theta_eq, bist_ang_az, eq_azi, u_10, fetch, polarization)
    return M * Br