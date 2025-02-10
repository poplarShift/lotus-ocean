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
from ltc.util.param import ParamUnfreezer

pn.extension()
hv.extension("bokeh")


ustar0_start = 0.005
ustar0_widget = pn.widgets.FloatSlider(
    name="Ice-ocean interface friction velocity (ustar0)",
    start=0.001,
    end=0.02,
    step=0.001,
    value=ustar0_start,
    format=PrintfTickFormatter(format="%.4f m/s"),
)

z0_start = 3e-2
z0_widget = pn.widgets.FloatSlider(
    name="z0",
    start=1e-4,
    end=0.1,
    step=5e-4,
    value=z0_start,
    format=PrintfTickFormatter(format="%.3f m"),
)


def get_model(ustar0=ustar0_start, z0=z0_start):

    data = pd.read_csv(top_level / "data/TS_stable.csv")
    data["CT"] = (
        data["CT"].rolling(window=5, center=True).mean().interpolate(method="bfill")
    )

    model = SteadyLTCModel.from_ctd(
        time=0,
        latitude=-70,
        SA=data["SA"],
        CT=data["CT"],
        zz=data["zz"],
        z0=z0,
        Sice=0,
        ustar0=ustar0,
        wb0=0,
        hice=10,
        qcond=0,
        params_dynamic=["wb0"],
    )

    model.steady_ltc()
    # print('z0 at ', model.z0)
    # with ParamUnfreezer(model):
    #     if Uice is not None:
    #         model.Uice = Uice
    #     if z0 is not None:
    #         model.z0 = z0
    # model.init_ekman_spiral()
    # model.find_steady_k()
    model.update_interface()
    # # print('z0 at ', model.z0)
    print("z0 at ", model.z0)
    print("ustar0 at", model.ustar0)
    print("Uice at", model.Uice)
    return model


def _plot_profile(model):
    ds = model.output_state().isel(time=0).drop("time")
    T = ds["T"].hvplot(ylabel="Temperature (deg C)")
    S = ds["S"].hvplot(ylabel="Salinity (g kg-1)")
    Uice = hv.Scatter([(0, np.abs(model.Uice))])
    U = np.abs(ds["U"]).hvplot(ylabel="Speed (m s-1)", ylim=(-0.01, 0.4)) * Uice
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


def callback_Uice(target, event):
    print("------- Setting Uice to ", event.new)
    model = get_model(ustar0=event.new, z0=z0_widget.value)
    print("ustar0 now at ", abs(model.ustar0))
    target.object = _plot_profile(model)


def callback_z0(target, event):
    print("----- Setting z0 to ", event.new)
    model = get_model(z0=event.new, ustar0=ustar0_widget.value)
    print("ustar0 now at ", abs(model.ustar0))
    target.object = _plot_profile(model)


def make_app():
    model = get_model()
    plot_panel = pn.pane.HoloViews(_plot_profile(model))

    ustar0_widget.link(plot_panel, callbacks={"value": callback_Uice})

    z0_widget.link(plot_panel, callbacks={"value": callback_z0})

    app = pn.Column(ustar0_widget, z0_widget, plot_panel, width=900)
    return app


if pn.state.served:
    pn.extension(throttled=True, sizing_mode="stretch_width")
    make_app().servable()
