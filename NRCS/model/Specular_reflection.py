from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import numpy as np


def Specular_reflection(kr, theta, azimuth, u_10, fetch, spec_name='elfouhaily'):
    """
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :param phi_wind:
    :param spec_name:
    :return:
    """

    nphi = theta.shape[0]
    nazi = azimuth.shape[0]
    # wave number
    kbr = 2*kr*np.sin(theta)

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]

    phi = (np.arange(nphi) * np.pi / nphi).reshape(1, nphi) - np.pi / 2  # in radians wave direction relative to the wind

    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        SSkdir = specf(kbr.reshape((nphi, 1)), u_10, fetch) * spreadf(kbr.reshape((nphi, 1)), phi, u_10, fetch)/kbr.reshape(nphi, 1) # equation 45
    else:
        SSkdir = specf(kbr.reshape((nphi, 1)), u_10, fetch, phi) * kbr.reshape(nphi, 1) ** 4  # equation 45
    # # mean square slope in upwind direction
    Sup = np.trapz(kbr.reshape(nphi, 1)**3*np.cos(phi)**2*SSkdir, phi, axis=1)
    # integration over k<kd
    Sup = np.trapz(Sup[kbr < const.kd], kbr[kbr < const.kd])

    # mean square slope in crosswind direction
    # integration over phi
    Scr = np.trapz(kbr.reshape(nphi, 1)**3*np.sin(phi)**2*SSkdir, phi, axis=1)
    # integration over k<kd
    Scr = np.trapz(Scr[kbr < const.kd], kbr[kbr < const.kd])

    # Kudryavtsev 2005 equation (10) comment
    # Mean squre slope satisfying conditions of the specular reflections
    Ssp = Sup * Scr / (Sup * np.sin(azimuth.reshape(nazi, 1)) ** 2 + Scr * np.cos(azimuth.reshape(nazi, 1)) ** 2)

    theta_ll = 0 # local incidence angle
    #  from Introduction to microwave remote sensing
    cos_theta_ll = np.cos(theta_ll)
    sqrt_e_sin = np.sqrt(const.epsilon_sw + cos_theta_ll**2. - 1.)
    RR_vv = ((const.epsilon_sw*cos_theta_ll - sqrt_e_sin)/(const.epsilon_sw*cos_theta_ll + sqrt_e_sin)) # equation [5.54]
    RR = RR_vv
    SP = np.pi*np.abs(RR)**2*np.exp(-np.tan(theta.reshape((1, nphi)))**2/(2*Ssp))/(2*np.pi*np.sqrt(Sup)*np.sqrt(Scr)*np.cos(theta.reshape(1, nphi))**4) # equation [10] in 2005
    return SP
