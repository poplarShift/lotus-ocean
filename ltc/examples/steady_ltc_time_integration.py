import pandas as pd, numpy as np, param

from ltc.run.run import ModelRun
from ltc.run.stepper import TempSalStepper
from ltc.run.forcing import Forcing, Timeseries
from ltc.iobl import IceOceanBoundaryLayer, LTCModel
from ltc.iobl.util import get_transfer_coefficients

data = pd.read_csv('data/TS.csv')

state = IceOceanBoundaryLayer.from_ctd(
    time=0, latitude=78,
    SA=data['SA'],
    CT=data['CT'],
    zz=data.zz,
    z0=0.049,
    Sice=4,
    ustar0=0.012,
    wb0=0,
    hice=2,
    qcond=0,
)

model = LTCModel(
    latitude=78,
    cht=0.0093,
    chs=0.0057,
    adjustments=[get_transfer_coefficients],
    T_btm_bc_kind='flux',
    T_btm_bc=0,
    S_btm_bc_kind='flux',
    S_btm_bc=0,
)

stepper = TempSalStepper(
    deltat=6,
    leapfrog_smoother=0.1,
)

forcing = Forcing(
    time=np.r_[0, np.cumsum(np.ones(1000-1)/4)],
#     ustar0=np.r_[.01*np.ones(12), 0.001*np.ones(1000)],
    ustar0=0.025*np.ones(1000),
    qcond=lambda state: 1e-6
)

scenario = ModelRun(
    model=model, state=state, stepper=stepper, forcing=forcing,
    write_output_step=1, duration=10,
)
scenario.steady_ltc()
scenario.state.init_leapfrog()

with param.logging_level('WARNING'):
    scenario.execute()
