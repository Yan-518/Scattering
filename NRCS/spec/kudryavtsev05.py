import numpy as np
from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import warnings
warnings.filterwarnings("ignore")

def fv(u_10):
    cg = np.sqrt((0.8 + 0.065*u_10) * 1e-3)  # drag coefficients
    U = cg * u_10  # friction velocity
    return U

def rough(u_10):
    return 0.018 * fv(u_10) ** 2 / const.g + 0.1 * const.v_air / fv(u_10)

def spec_peak(u_10, fetch):
    X_0 = 22e3  # Dimensionless fetch
    k_0 = const.g / u_10 ** 2
    # Eq. 4 (below)
    X = k_0 * fetch
    # Eq. 37: Inverse wave age
    Omega_c = 0.84 * np.tanh((X / X_0) ** (0.4)) ** (-0.75)
    # Eq. 3 (below)
    kp = k_0 * Omega_c ** 2
    return kp

def Cb(u_10, fetch):
    nk = 1024
    k_min = spec_peak(u_10, fetch)
    k = np.linspace(k_min, const.ky, nk)
    c = np.sqrt(const.g / k + const.gamma * k / const.rho_water)  # phase velocity
    U = fv(u_10)  # friction velocity
    Cb = 1.5 * (const.rho_air / const.rho_water) * (np.log(np.pi / (k * rough(u_10))) / const.kappa - c / U)
    Cb = np.where(Cb > 0, Cb, 0)
    Cb = np.mean(Cb[const.kwb <= k][k[const.kwb <= k] <= const.ky / 2])
    return Cb

def beta(k, u_10, phi):
    """
    Equation (17) in Kudryatvtsev 2005

    :param k:
    :param U:
    :param phi:
    :param z_0: Roughness scale, set to 0.0002 following Wikipedia
    :return:
    """

    U = fv(u_10)  # friction velocity
    # Phase velocity
    c = np.sqrt(const.g / k + const.gamma * k/const.rho_water)
    #
    c_b = 1.5 * (const.rho_air / const.rho_water) * (np.log(np.pi / (k * rough(u_10))) / const.kappa - c / U)
    c_b = np.where(c_b > 0, c_b, 0)
    # (17)
    return c_b * (U / c) ** 2 * np.cos(phi) * np.abs(np.cos(phi))

def beta_v(k, u_10, phi):
    """
    # After (16)
    :param k:
    :param u_10:
    :param phi:
    :param z_0:
    :param v: viscosity coefficient of sea water [m^2/s]
    :return:
    """
    omega = np.sqrt(const.g * k + const.gamma * k ** 3/const.rho_water)

    b_v = beta(k, u_10, phi) - 4 * const.v * k ** 2 / omega  # 2003a
    # b_v = beta(k, u_10, phi) - 4 * const.v * k ** 2 * np.cos(phi) * np.abs(np.cos(phi)) / omega  # 2003a

    return b_v

def crup_inc(phi):
    # based on experiemts before, assume >83 is downwind, <83 is crosswind
    cr_ed = np.cos(np.radians(83))
    dow_inc = np.where(np.cos(phi) > cr_ed)[0]
    cr_inc = np.where(np.logical_and(np.cos(phi) <= cr_ed, np.cos(phi) >= -cr_ed))[0]
    up_inc = np.where(np.cos(phi) < -cr_ed)[0]
    return cr_inc, up_inc, dow_inc

def func_f(k, k_num):
    return (1 + np.tanh(2 * (np.log(k) - np.log(k_num / 4)))) / 2

def param(k, u_10, fetch):
    #   computing tuning parameters
    n = []
    alpha = []
    ng = 5
    C_b = Cb(u_10, fetch)
    a = 2.5e-3
    f = func_f(k, const.ky)
    n = (1 - 1 / ng) * f + 1 / ng
    n = 1 / n
    alpha = np.log(a) - np.log(C_b) / n
    alpha = np.exp(alpha)
    return n, alpha

def B0(k, u_10, phi, fetch):
    # nk = k.shape[0]
    n, alpha = param(k, u_10, fetch)
    b_v = beta_v(k, u_10, phi)
    b_v = np.where(b_v > 0, b_v, 0)
    B_ref = alpha * b_v ** (1 / n)
    return B_ref

def B_wdir(k, u_10, phi, fetch):
    nk = k.shape[0]
    omega = np.sqrt(const.g * k + const.gamma * k ** 3/const.rho_water)
    n, alpha = param(k, u_10, fetch)
    B_ref = B0(k, u_10, phi, fetch)
    be = beta(k, u_10, phi)
    ins = abs(np.trapz(be * B_ref * omega / k, phi, axis = 1).reshape(nk, 1))
    SW = []
    for kk in k:
        km = min(kk/10, const.kwb)
        # SW.append(np.trapz(ins[k < km], k[k < km]))
        SW.append(np.trapz(np.sort(ins[k < km]), np.sort(k[k < km])))
    SW = const.cb * np.asarray(SW).reshape(nk, 1) / (2 * const.alphag * omega)
    # SW = 4.5e-3 * np.asarray(SW).reshape(nk, 1) / omega
    cr_inc, up_inc, dow_inc = crup_inc(phi)
    B_crw = alpha * (SW / alpha) ** (1 / (n+1)) * np.ones((nk, cr_inc.shape[0]))
    b_v = beta_v(k, u_10, phi)
    # B_upw = -SW / b_v[:, up_inc]
    B_upw = -SW / b_v
    B_crw = np.minimum(B_crw, B_upw[:, cr_inc])
    B_upw = B_upw[:, up_inc]
    return B_crw, B_upw

def B_dir(k, u_10, fetch, phi):
    b_v = beta_v(k, u_10, phi)
    B_ref = B0(k, u_10, phi, fetch)
    B_crw, B_upw = B_wdir(k, u_10, phi, fetch)
    Bk_h = b_v
    cr_inc, up_inc, dow_inc = crup_inc(phi)
    Bk_h[:, cr_inc] = B_crw
    Bk_h[:, dow_inc] = B_ref[:, dow_inc]
    Bk_h[:, up_inc] = B_upw
    return Bk_h

def B_ite(k, u_10, fetch, phi):
    dd = 1e-3
    nk = k.shape[0]
    BB0 = B_dir(k, u_10, fetch, phi)
    n, alpha = param(k.reshape(nk, 1), u_10, fetch)
    omega = np.sqrt(const.g * k + const.gamma * k ** 3 / const.rho_water)
    b_v = beta_v(k.reshape(nk, 1), u_10, phi)
    be = beta(k, u_10, phi)

    thres = 10
    while thres > 1e-5:
        ins = abs(np.trapz(be * BB0 * omega / k, phi, axis=1).reshape(nk, 1))
        dins = abs(np.trapz(be * (BB0 + dd) * omega / k, phi, axis=1).reshape(nk, 1))
        dSW = []
        SW = []
        for kk in k:
            km = min(kk / 10, const.kwb)
            # SW.append(np.trapz(ins[k < km], k[k < km]))
            # dSW.append(np.trapz(dins[k < km], k[k < km]))
            SW.append(np.trapz(np.sort(ins[k < km]), np.sort(k[k < km])))
            dSW.append(np.trapz(np.sort(dins[k < km]), np.sort(k[k < km])))
        SW = const.cb * np.asarray(SW).reshape(nk, 1) / (2 * const.alphag * omega)
        # SW = 4.5e-3 * np.asarray(SW).reshape(nk, 1) / omega
        dSW = const.cb * np.asarray(dSW).reshape(nk, 1) / (2 * const.alphag * omega)
        # dSW = 4.5e-3 * np.asarray(dSW).reshape(nk, 1) / omega

        Q = b_v * BB0 - BB0 * (BB0 / alpha) ** n + SW

        dQ = ((b_v * (BB0 + dd) - (BB0 + dd) * ((BB0 + dd) / alpha) ** n + dSW) - Q) / dd

        thres = np.abs(Q / dQ).max()
        # print(thres)

        BB0 = BB0 - Q / dQ

        BB0 = np.where(BB0 > 0, BB0, 0)

    return BB0

def Bpc(k, u_10, fetch, phi):
    nk = k.shape[0]
    kh = const.ky**2/const.kwb
    kl = 2*const.ky
    func_phi = func_f(k, kl)-func_f(k, kh)
    kg = const.ky**2/k.reshape(nk,1)
    bekg = beta(kg, u_10, phi)
    bekg = np.where(bekg > 0, bekg, 0)
    kg = np.flipud(kg)
    B0kg = B_ite(kg, u_10, fetch, phi)
    B0kg = np.flipud(B0kg)
    Ipc = bekg * B0kg * func_phi
          # /func_phi.max()
    n, alpha = param(k.reshape(nk, 1), u_10, fetch)
    omega = np.sqrt(const.g * k + const.gamma * k ** 3/const.rho_water)
    Bpcc = alpha * (-4 * const.v * k ** 2 / omega + np.sqrt((4 * const.v * k ** 2 / omega)**2 + 4 * Ipc / alpha)) / 2
    return Bpcc

def kudryavtsev05(k, u_10, fetch, phi):
    """
    :param k: wave number
    :param u_10: wind velocity at the height of 10 m above water
    :param phi: wave direction relative to the wind
    :return: background spectrum
    """
    nk = k.shape[0]
    k_p = spec_peak(u_10, fetch)
    c_p = np.sqrt(const.g / k_p + const.gamma * k_p / const.rho_water)
    Omega = u_10 / c_p
    cut = 1 - np.exp(-Omega / np.sqrt(10.) * (np.sqrt(k / k_p) - 1.))
    BK_h = B_ite(k, u_10, fetch, phi) + Bpc(k, u_10, fetch, phi)
    #
    # L_pm = np.exp(-5. / 4. * (spec_peak(u_10, fetch) / k) ** 2)
    # F_m = L_pm * np.exp(-0.25 * (k / const.ky - 1.) ** 2)

    # Omni directional spectrum model name
    specf = spec.models['elfouhaily']
    # Directional spectrum model name
    spreadf = spread.models['elfouhaily']

    B_l, B_h, Sk = specf(k.reshape(nk, 1), u_10, fetch, return_components='True')
    BE_l = B_l.reshape(nk, 1) * spreadf(k.reshape(nk, 1), phi, u_10, fetch)
    return BE_l + cut * BK_h