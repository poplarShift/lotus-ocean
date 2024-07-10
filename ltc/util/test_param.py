import pytest
import param
from ltc.util.param import ParamWatchdog, ParamForcer, ValueSetLogger


class Dummy(ParamWatchdog):
    a = param.Number()
    b = param.Number()


def test_paramwatchdog():
    p = Dummy(watch_param_set_events=True, a=99, params_forced=["a"])
    with pytest.raises(Exception):
        p.a = 123

    p = Dummy(watch_param_set_events=True, a=99, params_forced=[])
    p.a = 123

def test_params_forced_default():
    p = Dummy(watch_param_set_events=True, a=99)
    assert p.params_forced == ['a']
    with pytest.raises(Exception):
        p.a = 123

def test_param_forcer():
    p = Dummy(watch_param_set_events=True, a=99, params_forced=["a"])
    with ParamForcer(p):
        p.a = 123

def test_value_set_logger():
    p = Dummy(watch_param_set_events=True, a=99, params_forced=[])
    with ValueSetLogger(p) as vl:
        ev0 = vl.events
        p.a = 9
        ev1 = vl.events

    assert ev0 == []
    assert {'a'} == {e.param for e in ev1}

    with ValueSetLogger(p) as vl:
        ev0 = vl.events
        p.b = 99
        ev1 = vl.events

    assert ev0 == []
    assert {'b'} == {e.param for e in ev1}