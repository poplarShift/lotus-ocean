from ltc.iobl import IceOceanInterface
from ltc.iobl.util import get_transfer_coefficients
from ltc.util.param import ValueSetLogger


def test_update_interface():
    model = IceOceanInterface(
        time=0,
        latitude=-70,
        z0=0.01,
        Sice=0,
        Sff=35,
        Tff=-1,
        ustar0=0.0019,
        wb0=1.5e-8,
        qcond=0,
        watch_param_set_events=True,
        cht=0.0093,
        chs=0.0057,
        params_dynamic=["wb0"],
    )

    with ValueSetLogger(model) as vl:
        model.update_interface(update_wb0=True)
        ev = vl.events

    assert {"S0", "T0", "wS0", "wT0", "wb0", "w_interface"} == {e.param for e in ev}
