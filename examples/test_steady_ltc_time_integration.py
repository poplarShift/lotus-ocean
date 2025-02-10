import pytest
from steady_ltc_time_integration import setup, run, plot

def test_simulation_run():
    scenario = setup()
    run(scenario)
    ds = scenario.states
    plot(ds)
