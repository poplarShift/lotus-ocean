"""
The infrastructure for time integrations. Makes the following components
play together: stepper, model, and forcing.
"""

from copy import deepcopy
import warnings

import param
import numpy as np

from .stepper import Stepper
from .forcing import Forcing

try:
    import xarray as xr
except ImportError:
    xr = None
    warnings.warn("xarray not available.")


class ModelRun(param.Parameterized):
    """
    Set up and execute a model run.

    The start time is given by the time of the
    initial model state.
    """

    stepper = param.ClassSelector(
        class_=Stepper,
        allow_None=False,
        doc="""
        """,
    )

    duration = param.Number(
        default=0,
        doc="""
        Duration of model run (days).""",
    )

    write_output_step = param.Integer(
        default=0,
        doc="""
        If >=1, write out state every nth step.
        Collected states can be accessed using the .states property.""",
    )

    def __init__(self, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        self._states = []
        # self.execute()

    @property
    def states(self):
        run = self
        if xr is None:
            return run._states
        else:
            return xr.concat(run._states, "time")

    def execute(self):
        """
        Run the model for a given number of days.
        """
        stepper = self.stepper
        forcing = stepper.forcing
        state = stepper.state

        self._states = []
        stop_time = state.time + self.duration
        itcount = 0

        if forcing is not None:
            forcing.force(state=state)

        while state.time < stop_time:

            if self.write_output_step and not itcount % self.write_output_step:
                self._write_output()

            stepper.step()

            itcount += 1

        self._write_output()

    def _write_output(self):
        self._states.append(self.stepper.state.output_state())
