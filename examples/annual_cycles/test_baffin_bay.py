import sys
try:
    from cProfile import Profile
except:
    Profile = None
from pathlib import Path

top_level = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(top_level))
from examples.annual_cycles.baffin_bay import (
    get_forcing,
    get_ctd,
    setup_scenario,
    run,
)
def profilethis(logfile: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            prof = Profile()
            retval = prof.runcall(func, *args, **kwargs)
            prof.dump_stats(logfile)
            return retval

        return wrapper

    return decorator


outdir = Path("out/examples/baffin_bay/")

outdir.mkdir(parents=True, exist_ok=True)


def test_baffin_bay():
    forcing_df = get_forcing()
    ctd = get_ctd(forcing_df)
    print(ctd.isel(time=0), forcing_df)
    scenario = setup_scenario(ctd.isel(time=0), forcing_df, duration=1, watch_param_set_events=True)
    if Profile:
        profilethis(outdir / "test_run.cprof")(run)(scenario)
    else:
        run(scenario)
