import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import fv
from NRCS.modulation.Spectrum import wind_exp, wn_dless_single, wn_exp

def const_Trans(k, K, u_10, fetch, azimuth, div):
    nk = k.shape[0]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless_single(k, u_10)
    K_hat = K * fv(u_10) ** 2 / const.g
    wind_exponent = wind_exp(k)
    mk= wn_exp(k.reshape(nk, 1), u_10, fetch, azimuth)[:, 0]
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10) * div[:, :, None] / (const.g*(
                1 + 1j * c_tau * k_hat ** (-2) * K_hat[:, :, None]))
    return T

def const_spec(k, K, u_10, fetch, azimuth, div, wind_dir):
    nk = k.shape[2]
    T = const_Trans(k, K, u_10, fetch, azimuth, div)
    ind = np.where(np.degrees(azimuth) == wind_dir)[0]
    B_old = kudryavtsev05(k.reshape(nk, 1), u_10, fetch, azimuth)[:, ind][:, 0]
    B_new = B_old * (1 + np.abs(T))
    return B_old, B_new
