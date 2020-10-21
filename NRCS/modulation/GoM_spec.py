import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import fv
from NRCS.modulation.Spectrum import wind_exp, wn_dless, wn_exp

def GoM_Trans(k, K, u_10, fetch, azimuth, div):
    nk = k.shape[2]
    c_beta = 0.04  # wind wave growth parameter
    k_hat = wn_dless(k, u_10)
    K_hat = K * fv(u_10) ** 2 / const.g
    wind_exponent = np.zeros([div.shape[0], div.shape[1], nk])
    mk = np.zeros([div.shape[0], div.shape[1], nk])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            wind_exponent[ii ,jj, :] = wind_exp(k[ii, jj, :])
            mk[ii, jj, :] = wn_exp(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)[:, 0]
    c_tau = wind_exponent / (2 * c_beta)  # constant
    # transfer function
    T = c_tau * k_hat ** (-3 / 2) * mk * fv(u_10)[:, :, None] * div[:, :, None] / (const.g*(
                1 + 1j * c_tau * k_hat ** (-2) * K_hat[:, :, None]))
    return T

def GoM_spec(k, K, u_10, fetch, azimuth, div, wind_dir):
    nk = k.shape[2]
    T = GoM_Trans(k, K, u_10, fetch, azimuth, div)
    ind = np.where(np.degrees(azimuth) == wind_dir)[0]
    B_old = np.zeros([div.shape[0], div.shape[1], nk])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            B_old[ii, jj, :] = kudryavtsev05(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)[:, ind][:, 0]
    B_new = B_old * (1 + np.abs(T))
    return B_old, B_new