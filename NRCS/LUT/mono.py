import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak
import NRCS.constants as const
# from NRCS.model import Specular_reflection
from NRCS.model import Wave_breaking
from NRCS.model import Bragg_scattering

class nrcs_mono():
    """ :param kr: Radar wave number
        :param theta: Normal incidence angle vector
        :param azimuth: radar look angle relative to wind direction vector
        :param pol: Polarization ('vv', 'hh')
        :param u_10: Wind speed (U_10)
        :param fetch: Wind fetch
        :param spec_name: spectrum function
    """

    def __init__(self, kr, theta, azimuth, pol, u_10, fetch, spec_name):

        # Save parameters
        self.u_10 = u_10
        self.fetch = fetch
        self.pol = pol
        self.kr = kr
        self.theta = theta
        self.azimuth = azimuth
        self.spec_name = spec_name

    def nrcs(self):
        """
            :param k: wave number
        """
        u_10 = np.linspace(3, 15, 13)
        k_min = spec_peak(u_10, self.fetch)
        nk = 1024
        k = np.linspace(10 * k_min, const.ky, nk)
        nphi = 28
        theta = np.radians(np.linspace(20, 47, nphi))  # 28
        nazi = 37
        azimuth = np.radians(np.linspace(-180, 180, nazi)) # 37
        ind = np.where(u_10 == self.u_10)[0]
        # sp = Specular_reflection(self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name)
        wb, q = Wave_breaking(self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name, polarization = self.pol)
        br = Bragg_scattering(k[:, int(ind)], self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name, polarization = self.pol)

        # nrcs = (br + sp) * (1 - q) + wb * q
        nrcs = br * (1 - q) + wb * q
        ind_azi = np.where(azimuth == self.azimuth)[0]
        ind_theta = np.where(theta == self.theta)[0]
        return nrcs[ind_azi, ind_theta]

def mono(kr, theta, azimuth, pol, u_10, fetch, spec_name):
    point = nrcs_mono(kr, theta, azimuth, pol, u_10, fetch, spec_name)
    return point.nrcs()

# vv = mono(2*np.pi / 5.6e-2, np.radians(35), np.radians(10), 'VV', 10, 500e+3, 'kudryavtsev05')
# hh = mono(2*np.pi / 5.6e-2, np.radians(35), np.radians(10), 'HH', 10, 500e+3, 'kudryavtsev05')
# vh = mono(2*np.pi / 5.6e-2, np.radians(35), np.radians(10), 'VH', 10, 500e+3, 'kudryavtsev05')
