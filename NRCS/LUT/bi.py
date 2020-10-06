import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak
import NRCS.constants as const
from NRCS.bistatic import Br_bi
from NRCS.bistatic import Wb_bi
import drama.geo as sargeo
import xarray as xr

def bi(kr, theta_i, theta_s, theta_eq, bist_ang_az, fetch, spec_name, u_10, azi, inc_polar, re_polar):
    """ :param kr: Radar wave number
        :param theta: Normal incidence angle vector
        :param azimuth: radar look angle relative to wind direction vector
        :param u_10: Wind speed (U_10)
    """
    k_min = spec_peak(u_10, fetch)
    nk = 1024
    k = np.linspace(10 * k_min, const.ky, nk)

    # azimuth angle with respect to the wind direction
    azimuth = bist_ang_az/2-azi

    pol = 'VV'
    if inc_polar == 'V':
        pol == 'VV'
    else:
        pol = 'HH'

    if re_polar == 'Bragg':
        wb, q = Wb_bi(kr, theta_i, theta_s, theta_eq, bist_ang_az, azimuth, u_10, fetch, spec_name, pol, inc_polar, re_polar)
        br = Br_bi(k, kr, theta_i, theta_s, theta_eq, bist_ang_az, azimuth, u_10, fetch, spec_name, pol, inc_polar)
        return br * (1 - q) + wb * q
    else:
        wb, q = Wb_bi(kr, theta_i, theta_s, theta_eq, bist_ang_az, azimuth, u_10, fetch, spec_name, pol, inc_polar, re_polar)
        wb_vh, q_vh = Wb_bi(kr, theta_i, theta_s, theta_eq, bist_ang_az, azimuth, u_10, fetch, spec_name, 'VH', inc_polar, re_polar)
        br = Br_bi(k, kr, theta_i, theta_s, theta_eq, bist_ang_az, azimuth, u_10, fetch, spec_name, 'VH', inc_polar)
        return br * (1 - q_vh) + wb * q +wb_vh * q_vh

# # input parameters
# kr = 2 * np.pi / 5.6e-2
# spec_name = 'kudryavtsev05'
# fetch = 500e+3
#
# dau = 350e3  # along-track seperation
# parfile = 'D:/TU Delft/Msc thesis/stereoid/PAR/Hrmny_2020_1.cfg'
# swth_bst = sargeo.SingleSwathBistatic(par_file=parfile, dau=dau)
#     # angles
# nphi = 28
# theta_i = np.linspace(20, 47, nphi)  # 28
# theta_s = np.degrees(swth_bst.inc2slave_inc(np.radians(theta_i)))
# bist_ang_az = np.degrees(swth_bst.inc2bist_ang_az(np.radians(theta_i)))
# eq_azi = bist_ang_az/2
#
# # equivalent incident angle
# V1 = np.vstack((np.sin(np.radians(theta_i)), np.zeros(theta_i.shape[0]), -np.cos(np.radians(theta_i))))
# V2 = np.vstack((np.sin(np.radians(theta_s)) * np.cos(np.radians(bist_ang_az)),
#                     np.sin(np.radians(theta_s)) * np.sin(np.radians(bist_ang_az)), -np.cos(np.radians(theta_s))))
# Vm = (V1 + V2) / 2
# Vm = Vm / np.linalg.norm(Vm, axis=0)
# theta_eq = np.degrees(np.arccos(-Vm[2, :]))
#
# eq_azi = np.radians(eq_azi)
# theta_i = np.radians(theta_i)
# theta_s = np.radians(theta_s)
# theta_eq = np.radians(theta_eq)
# bist_ang_az = np.radians(bist_ang_az)
#
#
#
# u_10 = np.linspace(3, 21, 19)
# in_pol = ['V', 'H']
# re_pol = ['Bragg', 'orth']
# nazi = 37
# azimuth = np.radians(np.linspace(-180, 180, nazi)) # 37
# nrcsbi = np.zeros([nazi, nphi, u_10.shape[0], len(in_pol), len(re_pol)])
# for mm in np.arange(len(re_pol)):
#     for rr in np.arange(len(in_pol)):
#         for nn in np.arange(u_10.shape[0]):
#             for aa in np.arange(nazi):
#                 nrcsbi[aa, :, nn, rr, mm] = bi(kr, theta_i, theta_s, theta_eq, bist_ang_az, fetch, spec_name, u_10[nn], azimuth[aa], in_pol[rr], re_pol[mm])
# #



# BI = xr.DataArray(nrcsbi, coords=[theta_i, u_10, in_pol, re_pol], dims=['theta','u_10','in_pol','re_pol'])
# path = 'd:/TU Delft/Msc Thesis/LUT/bi.nc'
# BI.to_netcdf(path)
# obj = xr.open_dataarray(path)
