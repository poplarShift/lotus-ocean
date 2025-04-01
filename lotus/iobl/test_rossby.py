from pathlib import Path
from numpy.testing import assert_allclose
from ltc.iobl.rossby import *
import matplotlib.pyplot as plt

top_level = Path(__file__).resolve().parent.parent.parent

chs = 0.0027
cht = 0.0093
grav = 9.82
kappa = 0.4
Lambdastar = 0.028
QL = 83.3
RC = 0.2
use_three_eqn = True


def test_rossby():
    # compare to McPhee Fig 4.6

    ustar0 = np.linspace(1e-3, 3e-2)

    U_neutral = rossby_similarity(
        ustar0=ustar0,
        wb0=0,
        z=0,
        latitude=75,
        z0=0.03,
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
    )

    plt.figure()
    plt.scatter(U_neutral, np.abs(ustar0**2 / U_neutral**2))
    plt.grid()
    plt.xlabel("U")
    plt.ylabel("C_D")
    plt.title("compare to McPhee Fig 4.6")
    plt.savefig(top_level / "out/tests/rossby/neutral.png")

    # compare to Randelhoff et al., JPO 2014, Fig 15

    ustar0 = np.linspace(1e-3, 1.5e-2)

    U_neutral = rossby_similarity(
        ustar0=ustar0,
        wb0=0,
        z=0,
        latitude=75,
        z0=0.01,
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
    )

    U_stable = rossby_similarity(
        ustar0=ustar0,
        wb0=1e-8,
        z=0,
        latitude=75,
        z0=0.01,
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
    )

    plt.figure()
    plt.scatter(ustar0, np.abs(ustar0**2 / U_neutral**2))
    plt.scatter(ustar0, np.abs(ustar0**2 / U_stable**2))
    plt.grid()
    plt.xlabel("u*0")
    plt.ylabel("C_D")
    plt.legend(["wb0=0", "wb0=1e-8 m2 s-3"])
    plt.title("compare to Randelhoff et al., JPO 2014, Fig 15")
    plt.savefig(top_level / "out/tests/rossby/stable.png")


def test_rossby_inverse():

    chs = 0.0027
    cht = 0.0093
    hice = 1
    latitude = -75
    qcond = 0
    Sice = 0
    use_three_eqn = True
    w_perc = 0
    z = 0
    z0 = 0.01

    U = np.linspace(0.01, 0.3)
    temperature = -1.7
    salinity = 34

    ustar0 = rossby_similarity_inverse(
        U=U,
        Tff=temperature,
        Sff=salinity,
        chs=chs,
        cht=cht,
        hice=hice,
        kappa=kappa,
        Lambdastar=Lambdastar,
        latitude=latitude,
        qcond=qcond,
        QL=QL,
        RC=RC,
        Sice=Sice,
        use_three_eqn=use_three_eqn,
        w_perc=w_perc,
        z=z,
        z0=z0,
    )
    wb0, _, _, _, _, _ = interface_fluxes(
        ustar0,
        Tff=temperature,
        Sff=salinity,
        chs=chs,
        cht=cht,
        grav=grav,
        hice=hice,
        qcond=qcond,
        QL=QL,
        Sice=Sice,
        use_three_eqn=use_three_eqn,
        w_perc=w_perc,
    )

    U_reverse = rossby_similarity(
        ustar0=ustar0,
        wb0=wb0,
        z=z,
        latitude=latitude,
        z0=z0,
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
    )

    assert_allclose(np.abs(U_reverse), U, rtol=1e-3)


def test_compare_sltc():
    import pandas as pd, numpy as np, param
    import xarray as xr
    from ltc.run.run import ModelRun
    from ltc.run.stepper import TempSalStepper
    from ltc.run.forcing import Forcing, Timeseries
    from ltc.iobl import SteadyLTCModel
    from ltc.iobl.util import get_transfer_coefficients
    from ltc.util.time import Time
    from ltc.util.param import ValueSetLogger

    ustar0 = np.linspace(1e-3, 1e-2, 10)

    z0 = 0.01
    latitude = -70
    zz = np.arange(0, 60) + 0.5
    salinity = 35 * np.ones_like(zz)
    temperature = -1.8 * np.ones_like(zz)

    def get_sltc(ustar0, wb0):
        model = SteadyLTCModel.from_ctd(
            time=0,
            latitude=latitude,
            SA=salinity,
            CT=temperature,
            zz=zz,
            z0=z0,
            Sice=0,
            ustar0=ustar0,
            wb0=wb0,
            hice=10,
            qcond=0,
            cht=0.0093,
            chs=0.0057,
            adjustments=[get_transfer_coefficients],
            T_btm_bc_kind="flux",
            T_btm_bc=0,
            S_btm_bc_kind="flux",
            S_btm_bc=0,
        )

        model.steady_ltc()
        return model.Uice

    def get_rossby(ustar0, wb0):
        return rossby_similarity(
            ustar0=ustar0,
            wb0=wb0,
            z=0,
            latitude=latitude,
            z0=z0,
            Lambdastar=SteadyLTCModel.lam_star,
            kappa=SteadyLTCModel.kappa,
            RC=SteadyLTCModel.RC,
        )

    U_neutral_sltc = np.array([get_sltc(u, 0) for u in ustar0])
    U_neutral_rossby = get_rossby(ustar0, 0)
    U_stable_sltc = np.array([get_sltc(u, 1e-8) for u in ustar0])
    U_stable_rossby = get_rossby(ustar0, 1e-8)

    plt.figure()
    plt.plot(ustar0, np.abs(ustar0**2 / U_neutral_sltc**2), ".b")
    plt.plot(ustar0, np.abs(ustar0**2 / U_neutral_rossby**2), ".r")
    plt.plot(ustar0, np.abs(ustar0**2 / U_stable_sltc**2), "xb")
    plt.plot(ustar0, np.abs(ustar0**2 / U_stable_rossby**2), "xr")
    plt.xlabel("ustar0")
    plt.ylabel("(ustar0/U)^2")
    plt.legend(["SLTC neutral", "Rossby neutral", "SLTC stable", "Rossby stable"])
    plt.savefig("out/tests/rossby/sltc-compare.png")
