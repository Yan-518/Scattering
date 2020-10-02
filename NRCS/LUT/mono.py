import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak
import NRCS.constants as const
from NRCS.model import Wave_breaking
from NRCS.model import Bragg_scattering
import xarray as xr

def mono(u_10, pol):
    """ :param kr: Radar wave number
        :param theta: Normal incidence angle vector
        :param azimuth: radar look angle relative to wind direction vector
        :param pol: Polarization ('vv', 'hh')
        :param u_10: Wind speed (U_10)
        :param fetch: Wind fetch
        :param spec_name: spectrum function
    """
    kr = 2*np.pi / 5.6e-2
    spec_name = 'kudryavtsev05'
    fetch = 500e+3
    k_min = spec_peak(u_10, fetch)
    nk = 1024
    k = np.linspace(10 * k_min, const.ky, nk)
    nphi = 28
    theta = np.radians(np.linspace(20, 47, nphi))  # 28
    nazi = 37
    azimuth = np.radians(np.linspace(-180, 180, nazi)) # 37
    wb, q = Wave_breaking(kr, theta, azimuth, u_10, fetch, spec_name, pol)
    br = Bragg_scattering(k, kr, theta, azimuth, u_10, fetch, spec_name, pol)
    nrcs = br * (1 - q) + wb * q
    return nrcs

# u_10 = np.linspace(3, 15, 13)
# pol=['VV','HH', 'VH']
# nphi = 28
# theta = np.radians(np.linspace(20, 47, nphi))  # 28
# nazi = 37
# azimuth = np.radians(np.linspace(-180, 180, nazi))  # 37
# nrcsmn = np.zeros([nazi, nphi, u_10.shape[0], len(pol)])
# for mm in np.arange(len(pol)):
#     for nn in np.arange(u_10.shape[0]):
#         nrcsmn[:, :, nn, mm] = mono(u_10[nn], pol[mm])
#
# MN = xr.DataArray(nrcsmn, coords=[azimuth, theta, u_10, pol], dims=['azi','theta','u_10','pol'])
# path = 'd:/TU Delft/Msc Thesis/LUT/mono.nc'
# MN.to_netcdf(path)
# obj = xr.open_dataarray(path)