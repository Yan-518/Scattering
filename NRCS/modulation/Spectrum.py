import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import fv

def sample_back(k, u_10, tsc):
    nk = k.shape[2]
    ax0 = np.linspace(0, tsc.shape[0], k.shape[0] + 1)
    ax1 = np.linspace(0, tsc.shape[1], k.shape[1] + 1)
    k_re = np.zeros([tsc.shape[0], tsc.shape[1], nk])
    wind = np.zeros([tsc.shape[0], tsc.shape[1]])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            k_re[int(ax0[ii]):int(ax0[ii + 1]), int(ax1[jj]):int(ax1[jj + 1]), :] = k[ii, jj, :] * np.ones(
                [int(tsc.shape[0] / k.shape[0]), int(tsc.shape[1] / k.shape[1]), nk])
            wind[int(ax0[ii]):int(ax0[ii + 1]), int(ax1[jj]):int(ax1[jj + 1])] = u_10[ii, jj] * np.ones(
                [int(tsc.shape[0] / k.shape[0]), int(tsc.shape[1] / k.shape[1])])
    return k_re, wind

def wind_exp(k):
    # m*
    # the wind exponent of the spectrum form Trokhimovski, 2000
    kmin = 3.62e+2  # rad/m
    m_sta = []
    for kk in k:
        if kk <= kmin:
            m_sta.append(1.1 * kk ** 0.742)
        else:
            m_sta.append(5.61 - 1.19 * kk + 0.118 * kk ** 2)
    return np.asarray(m_sta)

def wn_dless(k, u_10):
    #     dimensionless wave number of wind waves
    return k*fv(u_10)[:, :, None]**2/const.g

def wn_dless_single(k, u_10):
    #     dimensionless wave number of wind waves
    return k*fv(u_10)**2/const.g

def wn_exp(k, u_10, fetch, azimuth):
    # mk
    # wave number exponent of the omnidirirectional spectrum of the wave action
    nk = k.shape[0]
    omega = np.sqrt(const.g * k + const.gamma * k ** 3 / const.rho_water)  # intrinstic frequency from dispersion relation
    # Omni directional spectrum model name
    Sk = kudryavtsev05(k.reshape(nk, 1), u_10, fetch, azimuth) * k.reshape(nk, 1) ** 4 # equation 45
    Sk = np.trapz(Sk, azimuth, axis=1)
    N = omega.reshape(nk, 1) * Sk.reshape(nk, 1) / k.reshape(nk, 1) ** 2
    mk = k.reshape(nk, 1) * np.gradient(np.log(N[:, 0]), k[1, 0] - k[0, 0]).reshape(nk, 1)
    return mk

def Trans(k, K, u_10, fetch, azimuth, tsc):
    nk = k.shape[2]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless(k, u_10)
    K_hat = K * fv(u_10) ** 2 / const.g
    wind_exponent = np.zeros([tsc.shape[0], tsc.shape[1], nk])
    mk = np.zeros([tsc.shape[0], tsc.shape[1], nk])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            wind_exponent[ii ,jj, :] = wind_exp(k[ii, jj, :])
            mk[ii, jj, :] = wn_exp(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)[:, 0]
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # divergence of the sea surface current
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10)[:, :, None] * divergence[:, :, None] / (const.g*(
                1 + 1j * c_tau * k_hat ** (-2) * K_hat[:, :, None]))
    return T

def Trans_single(k, K, u_10, fetch, azimuth, divergence):
    nk = k.shape[0]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless_single(k, u_10)
    K_hat = K*fv(u_10)**2/const.g
    wind_exponent = wind_exp(k)
    mk = wn_exp(k.reshape(nk, 1), u_10, fetch, azimuth)
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10) * divergence / (const.g*(1 + 1j * c_tau * k_hat ** (-2) * K_hat))
    return T

def Spectrum(k, K, u_10, fetch, azimuth, tsc, wind_dir):
    nk = k.shape[2]
    T = Trans(k, K, u_10, fetch, azimuth, tsc)
    ind = np.where(np.degrees(azimuth) == wind_dir)[0]
    B_old = np.zeros([tsc.shape[0], tsc.shape[1], nk])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            B_old[ii, jj, :] = kudryavtsev05(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)[:, ind][:, 0]
    B_new = B_old * (1 + np.abs(T))
    return B_old, B_new
