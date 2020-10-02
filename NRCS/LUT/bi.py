import numpy as np
from NRCS.spec.kudryavtsev05 import spec_peak
import NRCS.constants as const
from NRCS.bistatic import Br_bi
from NRCS.bistatic import Wb_bi
import drama.geo as sargeo

class nrcs_bi():
    """ :param kr: Radar wave number
        :param theta: Normal incidence angle vector
        :param azimuth: radar look angle relative to wind direction vector
        :param pol: Polarization ('vv', 'hh')
        :param u_10: Wind speed (U_10)
        :param fetch: Wind fetch
        :param spec_name: spectrum function
    """

    def __init__(self, parfile, kr, theta_i, pol, u_10, fetch, spec_name, inc_polar, re_polar):

        # Save parameters
        self.parfile = parfile
        self.u_10 = u_10
        self.fetch = fetch
        self.pol = pol
        self.kr = kr
        self.theta_i = theta_i
        self.spec_name = spec_name
        self.inc_polar = inc_polar
        self.re_polar = re_polar

    def nrcs(self):
        """
            :param k: wave number
        """
        u_10 = np.linspace(3, 15, 13)
        k_min = spec_peak(u_10, self.fetch)
        nk = 1024
        k = np.linspace(10 * k_min, const.ky, nk)
        ind = np.where(u_10 == self.u_10)[0]

        dau = 350e3  # along-track seperation
        swth_bst = sargeo.SingleSwathBistatic(par_file=self.parfile, dau=dau)
        # angles
        nphi = 51
        theta_i = np.linspace(20, 45, nphi)
        theta_s = np.degrees(swth_bst.inc2slave_inc(np.radians(theta_i)))
        eq_azi = np.degrees(swth_bst.inc2bist_ang_az(np.radians(theta_i)))/2

        # equivalent incident angle
        V1 = np.vstack((np.sin(theta_i), np.zeros(theta_i.shape[0]), -np.cos(theta_i)))
        V2 = np.vstack((np.sin(theta_s) * np.cos(eq_azi*2),
                        np.sin(theta_s) * np.sin(eq_azi*2), -np.cos(theta_s)))
        Vm = (V1 + V2) / 2
        Vm = Vm / np.linalg.norm(Vm, axis=0)
        theta_eq = np.degrees(np.arccos(-Vm[2, :]))


        eq_azi = np.radians(eq_azi)
        theta_i = np.radians(theta_i)
        theta_s = np.radians(theta_s)
        theta_eq = np.radians(theta_eq)

        wb, q = Wb_bi(self.kr, theta_i, theta_s, theta_eq, eq_azi, self.u_10, self.fetch, self.spec_name, self.pol, self.inc_polar, self.re_polar)
        br = Br_bi(k[:, int(ind)], self.kr, theta_i, theta_s, theta_eq, eq_azi, self.u_10, self.fetch, self.spec_name, self.pol, self.inc_polar)

        # nrcs = (br + sp) * (1 - q) + wb * q
        nrcs = br * (1 - q) + wb * q
        ind_theta = np.where(theta_i == self.theta_i)[0]
        return nrcs[ind_theta]

def bi(parfile, kr, theta_i, pol, u_10, fetch, spec_name, inc_polar, re_polar):
    # re_polar == 'Bragg' or 'orth'
    point = nrcs_bi(parfile, kr, theta_i, pol, u_10, fetch, spec_name, inc_polar, re_polar)
    return point.nrcs()
#
# parfile = 'D:/TU Delft/Msc thesis/stereoid/PAR/Hrmny_2020_1.cfg'
# vv = bi(parfile, 2*np.pi / 5.6e-2, np.radians(40), 'VV', 10, 500e+3, 'kudryavtsev05', 'V', 'Bragg')


