import numpy as np
import NRCS.constants as const
from NRCS import spec
from NRCS import spread
from NRCS.modulation.Spectrum import Trans
from NRCS.spec.kudryavtsev05 import fv

def MSS_em(K, u_10, tsc):
    # Empirical mss from Eq.17 in [Kudryavtsev, 2012]
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    cs = 180
    U = fv(u_10)
    return -cs * divergence / (U * np.sqrt(const.ky * K))

def MSS_contrasts(k, K, u_10, fetch, azimuth, spec_name, tsc):
    nk = k.shape[2]
    nazi = azimuth.shape[0]
    T = Trans(k, K, u_10, fetch, azimuth, tsc)
    mss = np.zeros([k.shape[0], k.shape[1], nazi])
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        spreadf = spread.models[spec_name]
        for ii in np.arange(k.shape[0]):
            for jj in np.arange(k.shape[1]):
                B_old = specf(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch) * spreadf(k[ii, jj, :].reshape(nk, 1), azimuth, u_10[ii, jj], fetch) * k[ii, jj, :].reshape(nk,1) ** 3
                s2_old = np.trapz(B_old/k[ii, jj, :].reshape(nk,1), k[ii, jj, :], axis =0)
                s2_new = np.trapz(B_old*(1+np.abs(T[ii, jj, :]))/k[ii, jj, :].reshape(nk,1), k[ii, jj, :], axis =0)
                mss[ii, jj, :] = (s2_new - s2_old) / s2_old
    else:
        for ii in np.arange(k.shape[0]):
            for jj in np.arange(k.shape[1]):
                B_old = specf(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)
                s2_old = np.trapz(specf(k[ii, jj, :].reshape(nk, 1), u_10[ii, jj], fetch, azimuth)/k[ii, jj, :].reshape(nk,1), k[ii, jj, :], axis =0)
                s2_new = np.trapz(B_old * (1 + np.abs(T[ii, jj, :]).reshape(nk,1)) / k[ii, jj, :].reshape(nk, 1),
                                             k[ii, jj, :], axis=0)
                mss[ii, jj, :] = (s2_new - s2_old) / s2_old
    return mss

