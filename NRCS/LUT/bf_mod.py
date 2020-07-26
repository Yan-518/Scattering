import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak
import NRCS.constants as const
from NRCS.model import Specular_reflection
from NRCS.model import Wave_breaking
from NRCS.model import Bragg_scattering

class nrcs_bf():
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
        k_min = spec_peak(self.u_10, self.fetch)
        k = np.linspace(10 * k_min, const.ky, 1024)

        theta = np.linspace(1, 90, 90) * np.pi / 180  # 512
        azimuth = np.linspace(-180, 180, 19*2-1) * np.pi / 180 # 1024

        sp = Specular_reflection(self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name)
        wb, q = Wave_breaking(self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name, polarizaton = self.pol)
        br = Bragg_scattering(k, self.kr, theta, azimuth, self.u_10, self.fetch, spec_name = self.spec_name, polarization = self.pol)

        theta = theta.reshape(1, 90)
        nrcs = (br + sp) * (1 - q) + wb * q
        nrcs = nrcs[azimuth == self.azimuth]
        nrcs = nrcs[theta == self.theta]
        return nrcs

if __name__ == "__main__":
    point = nrcs_bf(2*np.pi / 5.6e-2, 10*np.pi/180, 0, 'vv', 10, 500e+3, 'kudryavtsev05')
    print(point.nrcs())
