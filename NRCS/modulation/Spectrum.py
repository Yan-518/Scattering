import numpy as np
from NRCS import constants as const
from NRCS import spec
from NRCS import spread
from NRCS.spec.kudryavtsev05 import fv

def wind_exp(k):
    # m*
    # the wind exponent of the spectrum form Trokhimovski, 2000
    kmin = 3.62e+2  # rad/m
    m_sta = k
    m_sta[k <= kmin] = 1.1 * k[k <= kmin] ** 0.742
    m_sta[k > kmin] = 5.61 - 1.19 * k[k > kmin] + 0.118 * k[k > kmin] ** 2
    return m_sta

def wn_dless(k, u_10):
    #     dimensionless wave number of wind waves
    return k*fv(u_10)[:, :, None]**2/const.g

def wn_dless_single(k, u_10):
    #     dimensionless wave number of wind waves
    return k*fv(u_10)**2/const.g

def wn_exp(k, u_10, fetch, azimuth, spec_name):
    # mk
    # wave number exponent of the omnidirirectional spectrum of the wave action
    nk = k.shape[0]
    omega = np.sqrt(const.g * k + const.gamma * k ** 3 / const.rho_water)  # intrinstic frequency from dispersion relation
    # Omni directional spectrum model name
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        Sk = specf(k.reshape(nk, 1), u_10, fetch)
    else:
        Sk = specf(k.reshape(nk, 1), u_10, fetch, azimuth) * k.reshape(nk, 1) ** 4 # equation 45
        Sk = np.trapz(Sk, azimuth, axis=1)
    N = omega.reshape(nk, 1) * Sk.reshape(nk, 1) / k.reshape(nk, 1) ** 2
    mk = k.reshape(nk, 1) * np.gradient(np.log(N[:, 0]), k[1, 0] - k[0, 0]).reshape(nk, 1)
    return mk

def Trans(k, K, u_10, fetch, azimuth, tsc):
    nk = k.shape[2]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless(k, u_10)
    K_hat = K*fv(u_10)**2/const.g
    wind_exponent = np.zeros([k.shape[0], k.shape[1], nk])
    mk = np.zeros([k.shape[0], k.shape[1], nk])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            wind_exponent[ii, jj, :] = wind_exp(k[ii, jj, :])
            mk[ii, jj, :] = wn_exp(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth, 'kudryavtsev05')[:, 0]
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # divergence of the sea surface current
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10)[:, :, None] * divergence[:, :, None] / (const.g*(
                1 + 1j * c_tau * k_hat ** (-2) * K_hat[:, :, None]))
    return T

def Trans_single(k, K, u_10, fetch, azimuth, spec_name, divergence):
    nk = k.shape[0]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless_single(k, u_10)
    K_hat = K*fv(u_10)**2/const.g
    wind_exponent = wind_exp(k)
    mk = wn_exp(k.reshape(nk, 1), u_10, fetch, azimuth, spec_name)
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10) * divergence / (const.g*(1 + 1j * c_tau * k_hat ** (-2) * K_hat))
    return T

def Spectrum(k, K, u_10, fetch, azimuth, spec_name, tsc):
    nk = k.shape[2]
    nazi = azimuth.shape[0]
    T = Trans(k, K, u_10, fetch, azimuth, tsc)
    B_old = np.zeros([k.shape[0], k.shape[1], nk, nazi])
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        spreadf = spread.models[spec_name]
        for ii in np.arange(k.shape[0]):
            for jj in np.arange(k.shape[1]):
                B_old[ii, jj, :, :] = specf(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch) * spreadf(k[ii, jj, :].reshape(nk, 1), azimuth, u_10[ii, jj], fetch) * k[ii, jj, :].reshape(nk,1) ** 3
    else:
        for ii in np.arange(k.shape[0]):
            for jj in np.arange(k.shape[1]):
                B_old[ii, jj, :, :] = specf(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)

    B_new = B_old*(1+np.abs(T))
    return B_old, B_new
