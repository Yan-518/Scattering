import numpy as np
import NRCS.constants as const
from NRCS.spec.kudryavtsev05 import fv
from NRCS.current.vorticity import set_nan, nearest_nan, mean_zeros

def divergence(sst, lat, lon, wind_speed, K, phi, phi_w, s):
    """
    :param sst: sea surface temperature
    :param wind_speed:
    :param K: wave number vector
    :param phi: the direction of wave number vector
    :param phi_w: the direction of wind velocity vector
    :param s: sign of Coriolis parameter
    :return:
    """
    # interpolate data
    sst = set_nan(sst)
    sst = nearest_nan(sst, lat, lon)
    sst = mean_zeros(sst)

    ## compute the divergence
    T = np.fft.fft2(sst) # sst in fourier space
    V = np.sqrt(const.rho_air/const.rho_water)*fv(wind_speed)  # friction velocity in the water

    diver = 1j * const.alpha * const.g * V * (s * np.sin(phi_w - phi) + 1j * const.yita ** (3 / 4) * const.nb ** 0.5 * V * K / abs(const.f)) * K ** 2 * T / (const.yita ** 0.25 * const.nb ** 0.5 * const.f ** 2)

    # apply fan filter for divergence to show the structures
    a = np.array(range(0, int(sst.shape[1] / 2)))
    b = np.array(range(-int(sst.shape[1] / 2), 0))
    n = np.append(a, b)
    a = np.array(range(0, int(sst.shape[0] / 2)))
    b = np.array(range(-int(sst.shape[0] / 2), 0))
    m = np.append(a, b)
    mf = np.sinc(m / 500)
    mf[abs(m) > 500] = 0
    nf = np.sinc(n / 500)
    nf[abs(n) > 500] = 0
    fan = mf.reshape(mf.shape[0], 1) * nf
    diver = fan * diver
    D = np.fft.ifft2(diver)
    return D
