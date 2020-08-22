"""
The infrastructure for time integrations. Makes the following components
play together: stepper, model, state, and forcing.
"""

from copy import deepcopy

import param
import numpy as np

from .stepper import Stepper
from .forcing import Forcing
from ..iobl.state import State
from ..iobl.model import Model

try:
    import xarray as xr
except ImportError:
    xr = None
    warning.warn('xarray not available.')

class ModelRun(param.Parameterized):
    """
    Set up and execute a model run.

    The start time is given by the time of the
    initial model state.
    """

    state = param.ClassSelector(State, allow_None=False, doc="""
        Physical variables determining the state of the turbulent oceanic boundary
        layer.""")
    model = param.ClassSelector(Model, allow_None=False, doc="""
        A model that calculates turbulence parameters for a given state.""")
    stepper = param.ClassSelector(Stepper, allow_None=False, doc="""
        """)
    forcing = param.ClassSelector(Forcing, default=None, doc="""""")

    duration = param.Number(default=0, doc="""
        Duration of model run (days).""")

    write_output_step = param.Integer(default=0, doc="""
        If >=1, write out state every nth step.
        Collected states can be accessed using the .states property.""")


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
            return xr.concat(run._states, 'time')


    def execute(self):
        """
        Run the model for a given number of days.
        """

        model   = self.model
        state   = self.state
        forcing = self.forcing
        stepper = self.stepper

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
        self._states.append(self.state.output_state())

###

class KalmanFilterRun(ModelRun):

    def __init__(self, *argv, **kwarg):
        self._saved_filters = []
        super().__init__(*argv, **kwarg)


    def _write_output(self):
        super()._write_output()
        self._saved_filters.append(deepcopy(self.stepper.output_filter_params()))

    @property
    def saved_filters(model):
        if xr is None:
            return model._saved_filters
        else:
            ds = xr.concat(model._saved_filters, 'time')
            return ds
