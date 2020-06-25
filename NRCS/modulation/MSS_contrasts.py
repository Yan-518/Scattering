import numpy as np
from NRCS.modulation import Spectrum

def MSS(k, K, u_10, fetch, azimuth, spec_name, tsc):
    B_old, B_new = Spectrum(k, K, u_10, fetch, azimuth, spec_name, tsc)
    s2_new = np.trapz(B_new / k, k, axis=2)
    s2_old = np.trapz(B_old / k, k, axis=2)
    return (s2_new - s2_old) / s2_old
