#!/usr/bin/env python

import sys
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv

hv.extension("bokeh")
import gsw

import xarray as xr
import hvplot.xarray

top_level = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(top_level))

from ltc.iobl.model import SteadyLTCModel
from ltc.iobl.util import get_transfer_coefficients
from ltc.iobl.util import get_mixed_layer_depth_N2lim, get_mixed_layer_depth_sigth_crit


def construct_profile(
    z=None,
    mld=25,
    S0=32,
    Sgrad_ml=None,
    Sgrad_b=None,
    pc_form="efold",
    pc_thickness=None,
    Sgrad_pc=None,
    Sgrad_noise=0,
    Sb_scale=35,
):
    """
    Construct a synthetic CTD profile
    """
    from scipy.integrate import cumtrapz

    Sgrad = np.zeros_like(z, dtype=float)

    ix_ml = z < mld
    Sgrad[ix_ml] += Sgrad_ml

    if pc_form == "constgrad":
        ix_pc = (z >= mld) & (z < mld + pc_thickness)
        Sgrad[ix_pc] += Sgrad_pc

        ix_b = z >= mld + pc_thickness
        Sgrad[ix_b] = Sgrad_b

    elif pc_form == "efold":
        ix_pc = z >= mld
        Sgrad[ix_pc] = np.exp((mld - z[ix_pc]) / pc_thickness * Sgrad_pc)

    else:
        raise ValueError

    if Sgrad_noise is not None:
        Sgrad += Sgrad_noise * np.random.standard_normal(z.shape)

    if Sb_scale is not None:
        dS = np.sum(Sgrad)
        Sgrad /= dS / (Sb_scale - S0)

    S = S0 + np.r_[0, cumtrapz(Sgrad, z)]

    S = np.sort(S)
    T = np.zeros_like(S) + gsw.t_freezing(S0, 0, 1)

    df = pd.DataFrame(dict(CT=T, SA=S, zz=z))
    df["zz"] += df["zz"].diff()[1] / 2 - df["zz"][0]
    return df


def run_model(data, state_control_params):
    """
    Construct and run the model with the given CTD profile and state control parameters
    """

    model = SteadyLTCModel.from_ctd(
        adjustments=[get_transfer_coefficients],
        T_btm_bc_kind="flux",
        T_btm_bc=0,
        S_btm_bc_kind="flux",
        S_btm_bc=0,
        time=0,
        SA=data["SA"],
        CT=data["CT"],
        zz=data.zz,
        wb0=0,  # starting value
        watch_param_set_events=False,  # faster runs (no event watching overhead)
        **state_control_params,
    )

    # init state
    model.init_ekman_spiral()
    model.find_steady_k()

    return model.output_state().isel(time=0).drop_vars("time")


def permutations(params):
    """
    Gives list of dictionaries containing all permutations of key-value pairs in `params`
    """
    return [
        dict(i) for _, i in xr.Dataset(params).to_dataframe().reset_index().iterrows()
    ]


if __name__ == "__main__":
    ctd_control_params = dict(
        mld=[10, 20, 40],
        S0=34,
        Sgrad_ml=0.005,
        Sgrad_pc=0.4,
        pc_thickness=5,
        Sb_scale=35,
    )

    model_control_params = dict(
        latitude=[80],
        z0=[0.04, 0.004],
        Sice=[4],
        qcond=10.0 ** np.arange(-9, -6, 2),
        ustar0=np.arange(0.002, 0.02, 0.004),
        hice=[2],
    )

    results_ds = []
    param_names = list({**ctd_control_params, **model_control_params}.keys())

    for mld_alg_name, mld_alg in [
        ("sigcrit_0.03", partial(get_mixed_layer_depth_sigth_crit, sigth_crit=0.03)),
        ("sigcrit_0.3", partial(get_mixed_layer_depth_sigth_crit, sigth_crit=0.3)),
        ("N2lim", get_mixed_layer_depth_N2lim),
    ]:
        results = []
        for cparams in permutations(ctd_control_params):
            data = construct_profile(z=np.arange(0, 80), pc_form="efold", **cparams)
            for sparams in permutations(model_control_params):
                # with param.logging_level('WARNING'):
                ds = run_model(data, dict(**sparams, mld_algorithm=mld_alg))

                ds = ds.assign_coords(cparams)
                ds = ds.set_coords(param_names)
                ds = ds.expand_dims(param_names)
                results.append(ds)

        results_ds.append(
            xr.combine_by_coords(results).expand_dims(mld_alg=[mld_alg_name])
        )

    ds = xr.concat(results_ds, dim="mld_alg")

    e = ds.squeeze().Kh.hvplot("z", by="mld_alg", logy=True)
    hv.save(e, "out/examples/parameter_sensitivity.html")
