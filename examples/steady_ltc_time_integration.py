import sys, os

cwd = os.path.dirname(__file__)
ltc_base_path = os.path.join(cwd, "../")
if ltc_base_path not in sys.path:
    sys.path.insert(0, ltc_base_path)

import pandas as pd, numpy as np, param

from lotus.run.run import ModelRun
from lotus.run.stepper import TempSalStepper
from lotus.run.forcing import Forcing
from lotus.iobl import SteadyLTCModel
from lotus.iobl.util import get_transfer_coefficients
import matplotlib.pyplot as plt


def setup():
    data = pd.read_csv("data/TS.csv")

    state = SteadyLTCModel.from_ctd(
        time=0,
        latitude=78,
        SA=data["SA"],
        CT=data["CT"],
        zz=data.zz,
        z0=0.049,
        Sice=4,
        ustar0=0.012,
        wb0=0,
        hice=2,
        qcond=0,
        cht=0.0093,
        chs=0.0057,
        adjustments=[get_transfer_coefficients],
        T_btm_bc_kind="flux",
        T_btm_bc=0,
        S_btm_bc_kind="flux",
        S_btm_bc=0,
        params_dynamic=["wb0"],
    )

    forcing = Forcing(
        time=np.r_[0, np.cumsum(np.ones(4 - 1) / 4)],
        ustar0=0.025 * np.ones(4),
        qcond=lambda state: 1e-6,
    )

    stepper = TempSalStepper(
        state=state,
        forcing=forcing,
        deltat=6,
        leapfrog_smoother=0.1,
    )

    state.steady_ltc()
    state.init_leapfrog()

    scenario = ModelRun(
        stepper=stepper,
        write_output_step=1,
        duration=3,
    )
    return scenario


def run(scenario):
    with param.logging_level("WARNING"):
        scenario.execute()


def plot(ds):
    fig, ax = plt.subplots()
    ds.T.plot(center=False, ax=ax)
    ax.invert_yaxis()
    ds.mld.plot(ax=ax, c="k")
    fig.savefig("out/tests/run/sltc_example.png")


if __name__ == "__main__":
    scenario = setup()
    run(scenario)
    ds = scenario.states
    plot(ds)
