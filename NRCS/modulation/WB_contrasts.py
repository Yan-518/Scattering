import numpy as np
import NRCS.constants as const
from NRCS.spec.kudryavtsev05 import fv

def WB_contrasts(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc):
    # empirical wave breaking contrasts from Eq.17 in [Kudryavtsev, 2012]

    cq = 470
    U = fv(u_10)
    kb = kr / 10
    omegab = np.sqrt(const.g * kb + const.gamma * kb ** 3/const.rho_water)
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    wb_c = - cq * np.log(U * kb / np.sqrt(const.g * K)) * const.g * divergence / (U**2 * kb * omegab)
    return wb_c