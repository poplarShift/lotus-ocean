from pathlib import Path
from collections import defaultdict

top_level = Path(__file__).resolve().parent.parent.parent

import pandas as pd, numpy as np, param
import xarray as xr
import gsw
import matplotlib.pyplot as plt

from ltc.iobl import SteadyLTCModel
from ltc.iobl.util import get_transfer_coefficients
from ltc.util.param import ValueSetLogger


def test_sltc():
    data = pd.read_csv(top_level / "data/TS.csv")

    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=data["SA"],
        CT=data["CT"],
        zz=data["zz"],
        z0=0.01,
        Sice=0,
        ustar0=0.0019,
        wb0=1.5e-8,
        qcond=0,
        watch_param_set_events=True,
        cht=0.0093,
        chs=0.0057,
        T_btm_bc_kind="flux",
        T_btm_bc=0,
        S_btm_bc_kind="flux",
        S_btm_bc=0,
    )
    model.steady_ltc()


def test_fix_ice_velocity():
    zz = np.arange(0, 60) + 0.5
    salinity = 35 * np.ones_like(zz)
    temperature = -1.8 * np.ones_like(zz)
    salinity = gsw.SA_from_SP(salinity, zz, 0, -70)
    temperature = gsw.CT_from_pt(salinity, temperature)
    speed = 0.1

    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=salinity,
        CT=temperature,
        zz=zz,
        z0=0.025,
        Sice=0,
        Uice=speed,
        wb0=0,
        hice=10,
        qcond=0,
    )

    for i in range(3):
        model.steady_ltc()
    assert abs(speed / model.Uice - 1) < 5e-3


def test_z0_vs_ustar0_for_given_Uice():
    zz = np.arange(0, 60) + 0.5
    salinity = 35 * np.ones_like(zz)
    temperature = -1.8 * np.ones_like(zz)
    salinity = gsw.SA_from_SP(salinity, zz, 0, -70)
    temperature = gsw.CT_from_pt(salinity, temperature)

    def get_u0(z0, Uice):
        model = SteadyLTCModel.from_ctd(
            time=0,
            latitude=-70,
            SA=salinity,
            CT=temperature,
            zz=zz,
            z0=z0,
            Sice=0,
            Uice=Uice,
            wb0=0,
            hice=10,
            qcond=0,
        )
        model.steady_ltc()
        return model.ustar0, model.U[0]

    z0 = np.arange(1e-4, 0.1, 5e-3)
    Uice = [0.01, 0.05, 0.1, 0.2]
    ustar0 = defaultdict(lambda: np.ones_like(z0, dtype=complex))
    U0 = defaultdict(lambda: np.ones_like(z0, dtype=complex))
    for i, z in enumerate(z0):
        for U in Uice:
            ustar0[U][i], U0[U][i] = get_u0(z, U)

    plt.figure()
    for U in Uice:
        plt.plot(z0, np.abs(ustar0[U]), ".-")
    plt.xlabel("z0 (m)")
    plt.ylabel("abs(ustar0) (m s-1)")
    plt.grid()
    plt.title(f"ustar0 vs z0")
    plt.legend([f"Uice={U} m s-1" for U in Uice])
    plt.savefig("out/tests/iobl/ustar0-vs-z0-for-given-Uice.png")

    plt.figure()
    for U in Uice:
        plt.plot(z0, np.abs(U0[U]), ".-")
    plt.xlabel("z0 (m)")
    plt.ylabel("abs(U0) (m s-1)")
    plt.grid()
    plt.title(f"First grid cell velocity (U0) vs z0")
    plt.legend([f"Uice={U} m s-1" for U in Uice])
    plt.savefig("out/tests/iobl/U0-vs-z0-for-given-Uice.png")

    for U in Uice:
        assert np.all(np.abs(U0[U])<U)

def test_z0_vs_U0_for_given_ustar0():
    zz = np.arange(0, 60) + 0.5
    salinity = 35 * np.ones_like(zz)
    temperature = -1.8 * np.ones_like(zz)
    salinity = gsw.SA_from_SP(salinity, zz, 0, -70)
    temperature = gsw.CT_from_pt(salinity, temperature)

    def get_u0(z0, ustar0):
        model = SteadyLTCModel.from_ctd(
            time=0,
            latitude=-70,
            SA=salinity,
            CT=temperature,
            zz=zz,
            z0=z0,
            Sice=0,
            ustar0=ustar0,
            wb0=0,
            hice=10,
            qcond=0,
        )
        model.steady_ltc()
        return model.Uice, model.U[0]

    z0 = np.arange(1e-4, 0.1, 5e-3)
    ustar0 = [0.002, 0.005, 0.01, 0.02]
    Uice = defaultdict(lambda: np.ones_like(z0, dtype=complex))
    U0 = defaultdict(lambda: np.ones_like(z0, dtype=complex))
    for i, z in enumerate(z0):
        for u in ustar0:
            Uice[u][i], U0[u][i] = get_u0(z, u)

    plt.figure()
    for u in ustar0:
        plt.plot(z0, np.abs(Uice[u]), ".-")
    plt.xlabel("z0 (m)")
    plt.ylabel("abs(Uice) (m s-1)")
    plt.grid()
    plt.title(f"Uice vs z0")
    plt.legend([f"ustar0={u} m s-1" for u in ustar0])
    plt.savefig("out/tests/iobl/Uice-vs-z0-for-given-ustar0.png")

    plt.figure()
    for u in Uice:
        plt.plot(z0, np.abs(U0[u]), ".-")
    plt.xlabel("z0 (m)")
    plt.ylabel("abs(U0) (m s-1)")
    plt.grid()
    plt.title(f"First grid cell velocity (U0) vs z0")
    plt.legend([f"ustar0={U} m s-1" for U in ustar0])
    plt.savefig("out/tests/iobl/U0-vs-z0-for-given-ustar0.png")


def test_set_values():
    zz = np.arange(0, 60) + 0.5
    salinity = 35 * np.ones_like(zz)
    temperature = -1.8 * np.ones_like(zz)
    salinity = gsw.SA_from_SP(salinity, zz, 0, -70)
    temperature = gsw.CT_from_pt(salinity, temperature)
    speed = 0.1

    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=salinity,
        CT=temperature,
        zz=zz,
        z0=0.0025,
        Sice=0,
        Uice=speed,
        wb0=0,
        hice=10,
        qcond=0,
    )

    with ValueSetLogger(model) as vl:
        model.steady_ltc()
        ev = vl.events

    assert "wb0" not in {e.param for e in ev}
