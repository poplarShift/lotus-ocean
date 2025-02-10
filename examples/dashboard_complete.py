from pathlib import Path
from copy import copy

top_level = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(top_level))

from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from ltc.iobl import SteadyLTCModel
from ltc.iobl.util import get_transfer_coefficients, get_mixed_layer_depth_sigth_crit
from ltc.util.param import ParamProtectFrozen

import holoviews as hv

# hv.extension('matplotlib')
hv.extension("bokeh")
from holoviews import opts

import xarray as xr
import hvplot.xarray
import hvplot.pandas

import param
# param.get_logger().setLevel(param.DEBUG)
import panel as pn

pn.extension()

data = pd.read_csv(top_level / "data/TS.csv")

from param import depends


class Viewer(param.Parameterized):

    view_var = param.ObjectSelector(
        default="ustar",
        objects=SteadyLTCModel._z_grid + SteadyLTCModel._zz_grid,
        label="viewing var",
        precedence=0,
    )

    trigger = param.Action(lambda x: x.param.trigger("trigger"), label="Update")

    model_control_params = [
        "ustar0",
        "z0",
        "qcond",
        # "w_perc",
        # "latitude",
        "cht",
        "chs",
        # "km_min",
        # "kh_min",
        # "kappa",
        # "lam_star",
    ]
    model_diagnostics = ["wb0", "wb_pc"]

    def __init__(self, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        self._init_model()

    def _init_model(self):
        self.model = SteadyLTCModel.from_ctd(
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
            params_dynamic=copy(self.model_diagnostics),
            protect_frozen=False, # otherwise, we run into trouble when setting parameters using widgets
        )
        self.update_model()

    @depends(
        "trigger",
        #              *['model.'+p for p in model_control_params],
        #              *['model.state.'+p for p in state_control_params],
        watch=True,
    )
    def update_model(self):
        with ParamProtectFrozen(self.model):
            # This is more of a safety - we want to be sure that no parameter is
            # modified that should not
            self.model.init_ekman_spiral()
            self.model.find_steady_k()
            self.model.update_interface()
            self.model.finalize_interface_shear()
            self.ds = self.model.output_state().isel(time=0).drop("time")

    def controls(self):
        return pn.Column(
            pn.pane.Markdown("# Controls"),
            pn.Param(
                self,
                parameters=["view_var", "trigger"],
            ),
            pn.Row(
                pn.Param(self.model, parameters=self.model_control_params),
                # pn.Param(self.model.state, parameters=self.state_control_params, name='Forcing'),
            ),
        )

    def diagnostics(self):
        return pn.Column(
            pn.pane.Markdown("# Diagnostics"),
            pn.Row(
                pn.Param(
                    self.model, parameters=self.model_diagnostics, name="Diagnostics"
                ),
            ),
        )

    @staticmethod
    def _plot_profile(da):
        e = da.hvplot()
        e = e.opts(
            invert_axes=True,
            invert_yaxis=True,
        )
        return e

    @depends(
        "view_var",
        "trigger",
    )
    def view_profile(self):
        self.update_model()
        e = self._plot_profile(self.ds[self.view_var])
        return e.opts(
            frame_width=300,
            frame_height=400,
        )

    def view_ctd(self):
        l = self._plot_profile(self.ds.T) + self._plot_profile(self.ds.S)
        return l.opts(
            opts.Curve(
                frame_width=150,
                frame_height=200,
            )
        )

def make_app():
    p = Viewer()

    for c in p.model_control_params + p.model_diagnostics:
        setattr(getattr(p.model.param, c), "bounds", (-100, 100))

    # widgets = {'wb0': pn.widgets.LiteralInput, 'view_var': pn.widgets.LiteralInput}
    pn.Param._mapping[param.Number] = pn.widgets.LiteralInput

    app = pn.Row(
        p.controls,
        pn.Column(p.view_profile, p.view_ctd),
        p.diagnostics,
    )
    return app

if pn.state.served:
    pn.extension(sizing_mode="stretch_width")
    make_app().servable()

