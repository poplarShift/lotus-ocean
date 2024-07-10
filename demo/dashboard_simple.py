import sys
from pathlib import Path

import pandas as pd
from bokeh.models.formatters import PrintfTickFormatter
import holoviews as hv
import hvplot.xarray
import panel as pn
import numpy as np

top_level = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(top_level))

from ltc.iobl import SteadyLTCModel
from ltc.util.param import ParamForcer

pn.extension()
hv.extension("bokeh")
data = pd.read_csv(top_level / "data/TS_stable.csv")
data['CT'] = data['CT'].rolling(window=5, center=True).mean().interpolate(method='bfill')

Uice_start = 0.01
Uice_widget = pn.widgets.FloatSlider(
    name="Ice drift speed",
    start=0.01,
    end=0.3,
    step=0.01,
    value=Uice_start,
    format=PrintfTickFormatter(format="%.2f m/s"),
)

z0_start = 3e-2
z0_widget = pn.widgets.FloatSlider(
    name="z0",
    start=1e-4,
    end=0.5,
    step=5e-4,
    value=z0_start,
    format=PrintfTickFormatter(format="%.3f m"),
)


def get_model(Uice=Uice_start, z0=z0_start):
    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=data["SA"],
        CT=data["CT"],
        zz=data["zz"],
        z0=z0,
        Sice=0,
        Uice=Uice,
        wb0=1.5e-8,
        qcond=0,
        watch_param_set_events=True,
        cht=0.0093,
        chs=0.0057,
        T_btm_bc_kind="flux",
        T_btm_bc=0,
        S_btm_bc_kind="flux",
        S_btm_bc=0,
        params_dynamic=["wb0"],
    )
    # print('z0 at ', model.z0)
    # with ParamForcer(model):
    #     if Uice is not None:
    #         model.Uice = Uice
    #     if z0 is not None:
    #         model.z0 = z0
    model.init_ekman_spiral()
    model.find_steady_k()
    model.update_interface()
    # print('z0 at ', model.z0)
    model._update_interface_shear()
    # print('z0 at ', model.z0)
    return model


def _plot_profile(model):
    ds = model.output_state().isel(time=0).drop("time")
    T = ds["T"].hvplot(ylabel="Temperature (deg C)")
    S = ds["S"].hvplot(ylabel="Salinity (g kg-1)")
    Uice = hv.Scatter([(0, np.abs(model.Uice))])
    U = np.abs(ds["U"]).hvplot(ylabel="Speed (m s-1)", ylim=(-.01, 0.4)) * Uice
    Fh0 = hv.Scatter([(0, model.wT0 * 4e6)])
    Fh = (ds["Fh"].where(ds["Fh"] > 1e-3, 1e-5)).hvplot(
        ylim=(1e-3, 1e3),
        logx=True,
        ylabel="Upward heat flux (W m-2)",
    ) * Fh0
    l = T + S + U + Fh
    l = l.opts(
        hv.opts.Curve(
            invert_axes=True,
            invert_yaxis=True,
            frame_width=180,
            frame_height=500,
            xlabel="",
            color="k",
            show_grid=True,
        ),
        hv.opts.Scatter(
            color="r",
            size=10,
        ),
    )
    T = T.opts(
        hv.opts.Curve(
            xlabel="Depth (m)",
        ),
    )
    return l


model = get_model()
plot_panel = pn.pane.HoloViews(_plot_profile(model))


def callback_Uice(target, event):
    print("------- Setting Uice to ", event.new)
    model = get_model(Uice=event.new, z0=z0_widget.value)
    print("delU now at  ", abs(model.delU))
    print("ustar0 now at ", abs(model.ustar0))
    target.object = _plot_profile(model)


Uice_widget.link(plot_panel, callbacks={"value": callback_Uice})


def callback_z0(target, event):
    print("----- Setting z0 to ", event.new)
    model = get_model(z0=event.new, Uice=Uice_widget.value)
    print("delU now at  ", abs(model.delU))
    print("ustar0 now at ", abs(model.ustar0))
    target.object = _plot_profile(model)


z0_widget.link(plot_panel, callbacks={"value": callback_z0})


app = pn.Column(Uice_widget, plot_panel)
app.servable()
