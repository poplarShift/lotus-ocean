from pathlib import Path

top_level = Path(__file__).resolve().parent.parent.parent

import pandas as pd, numpy as np

# import xarray as xr
from ltc.run.run import ModelRun
from ltc.run.stepper import TempSalStepper
from ltc.run.forcing import Forcing
from ltc.iobl import SteadyLTCModel
from ltc.util.param import ValueSetLogger

data = pd.read_csv(top_level / "data/TS.csv")


def test_run():
    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=data["SA"],
        CT=data["CT"],
        zz=data.zz,
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
        params_dynamic=["wb0", "time"],
    )
    model.steady_ltc()
    model.init_leapfrog()

    forcing = Forcing(
        time=np.r_[0, np.cumsum(np.ones(1000 - 1) / 4)],
        ustar0=0.015 * np.ones(1000),
        qcond=lambda state: 1e-6,
    )

    stepper = TempSalStepper(
        deltat=6,
        leapfrog_smoother=0.1,
        state=model,
        forcing=forcing,
    )

    scenario = ModelRun(
        stepper=stepper,
        write_output_step=1,
        duration=1,
    )

    scenario.execute()
    ds = scenario.states
