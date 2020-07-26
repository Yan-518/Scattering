import numpy as np
import NRCS.constants as const
from NRCS.modulation import Spectrum
from NRCS.spec.kudryavtsev05 import fv

def MSS_em(K, u_10, tsc):
    # Empirical mss from Eq.17 in [Kudryavtsev, 2012]
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    cs = 180
    U = fv(u_10)
    return -cs * divergence / (U * np.sqrt(const.ky * K))

def MSS(k, K, u_10, fetch, azimuth, spec_name, tsc):
    B_old, B_new = Spectrum(k, K, u_10, fetch, azimuth, spec_name, tsc)
    s2_new = np.trapz(B_new / k, k, axis=2)
    s2_old = np.trapz(B_old / k, k, axis=2)
    return (s2_new - s2_old) / s2_old
