import numpy as np
import scipy.io as spio
from drama import constants as cnst
from drama import utils as drtls

def GoM(matfile, smp_out=None):
    """
    Read tsc and wind from mat file (in Claudia Pasquero's format)
    :param matfile:
    :return:
    """
    scn = spio.loadmat(matfile)
    excludes = ['__header__', '__version__', '__globals__']
    for key in scn:
        if key not in excludes:
            scn[key] = scn[key].transpose()
    tsc_v = np.zeros(scn['u_s'].shape + (2,))
    wind_v = np.zeros_like(tsc_v)
    tsc_v[:, :, 0] = scn['u_s']
    tsc_v[:, :, 1] = scn['v_s']
    # convert wind stress to wind speed with Cd=1e-3, air_density = 1.22 kg/m^3
    ind = scn['sustr_s'] < 0
    scn['sustr_s'][ind] = -scn['sustr_s'][ind]
    wind_v[:, :, 0] = np.sqrt(scn['sustr_s']/1.22e-3)
    wind_v[:, :, 0][ind] = -wind_v[:, :, 0][ind]
    ind = scn['svstr_s'] < 0
    scn['svstr_s'][ind] = -scn['svstr_s'][ind]
    wind_v[:, :, 1] = np.sqrt(scn['svstr_s']/1.22e-3)
    wind_v[:, :, 1][ind] = -wind_v[:, :, 1][ind]
#     zeta = scn['zeta_s']
    lat = scn['lat_s']
    lon = scn['lon_s']
    dx = np.radians(lon[0, 1] - lon[0, 0]) * cnst.r_earth
    dy = np.radians(lat[1, 0] - lat[0, 0]) * cnst.r_earth
    if smp_out is None:
        smp_out = dx
    else:
        # Resample
        nxo = int(np.floor(tsc_v.shape[1] * dx / smp_out))
        nyo = int(np.floor(tsc_v.shape[0] * dy / smp_out))
        xo = np.arange(nxo) * smp_out / dx
        yo = np.arange(nyo) * smp_out / dy
        wind_v = drtls.linresample(drtls.linresample(wind_v, xo, axis=1, extrapolate=True),
                                   yo, axis=0, extrapolate=True)
        tsc_v = drtls.linresample(drtls.linresample(tsc_v, xo, axis=1, extrapolate=True),
                                  yo, axis=0, extrapolate=True)
#         zeta = drtls.linresample(drtls.linresample(zeta, xo, axis=1, extrapolate=True),
#                                   yo, axis=0, extrapolate=True)

    return tsc_v, wind_v, smp_out