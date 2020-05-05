import numpy as np
from NRCS import constants as const
from NRCS import spec
from NRCS.spec.kudryavtsev05 import fv
def wind_exp(k):
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
    return k*fv(u_10)**2/const.g

def wn_exp(k, u_10, fetch, azimuth, spec_name):
    # wave number exponent of the omnidirirectional spectrum of the wave action
    nk = k.shape[0]
    omega = np.sqrt(const.g * k + const.gamma * k ** 3 / const.rho_water)  # intrinstic frequency from dispersion relation
    # Omni directional spectrum model name
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        Sk = specf(k.reshape(nk, 1), u_10, fetch)
        # spreadf = spread.models[spec_name]
        # Sk = Sk * spreadf(k.reshape(nk, 1), azimuth, u_10, fetch) / k.reshape(nk, 1)  # equation 45
    else:
        Sk = specf(k.reshape(nk, 1), u_10, fetch, azimuth) * k.reshape(nk, 1) ** 4 # equation 45
        Sk = np.trapz(Sk, azimuth, axis=1)
    N = omega.reshape(nk, 1) * Sk.reshape(nk, 1) / k.reshape(nk, 1) ** 2
    mk = k.reshape(nk, 1) * np.gradient(np.log(N[:, 0]), k[1, 0] - k[0, 0]).reshape(nk, 1)
    return mk

def Trans(k, K, u_10, fetch, azimuth, spec_name, tsc):
    nk = k.shape[0]
    c_beta = 0.04  # wind wave growth parameter
    c_tau = wind_exp(k) / (2 * c_beta)  # constant
    k_hat = wn_dless(k, u_10)
    K_hat = wn_dless(K, u_10)
    mk = wn_exp(k, u_10, fetch, azimuth, spec_name)
    # divergence of the sea surface current
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    # transfer function
    T = np.zeros((K.shape[0], K.shape[1], nk))
    for i in np.arange(nk):
        T[:, :, i] = c_tau[i] * k_hat[i] ** (-3/2) * mk[i] * fv(u_10) * divergence / (const.g * (1 + 1j * c_tau[i] * k_hat[i] ** (-2) * K_hat))
    return T

def Spectrum(k, K, u_10, fetch, azimuth, spec_name, tsc):
    nk = k.shape[0]
    T = Trans(k, K, u_10, fetch, azimuth, spec_name, tsc)
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        B = specf(k.reshape(nk, 1), u_10, fetch) * k.reshape(nk,1) ** 3
    else:
        B = specf(k.reshape(nk, 1), u_10, fetch, azimuth)
        B = np.trapz(B, azimuth, axis=1)
    B = B[:, 0] * (1 + np.abs(T))
    return B
