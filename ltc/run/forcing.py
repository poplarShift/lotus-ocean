"""
Time series of environmental forcing parameters.
"""
from collections.abc import Sequence
from copy import deepcopy
import param
import gsw
import numpy as np

from ..util.time import Time


class Timeseries(param.Parameterized):
    """
    Provide time series of observed values.

    Instantiate with either parameter-values or parameter-callable pairs
    """

    def __init__(self, *argv, **kwarg):
        self._callables = {}
        self._values_as_series = {} # {parameter: [series]}
        self._values = {} # {timestamp: {parameter: value}}

        self._validate_kwargs(kwarg)

        for param, value in kwarg.items():
            if callable(value):
                self._callables[param] = value
            else:
                self._values_as_series[param] = value
            setattr(self, param, value)

        self._prepare_value_forcing()


    def __getitem__(self, time):
        try:
            return {**self._values[time], **self._callables}
        except KeyError:
            try:
                return {**self._values[time.value], **self._callables}
            except KeyError:
                return None

    @staticmethod
    def _validate_kwargs(kwarg):
        if 'time' not in kwarg:
            raise ValueError('Need to supply time stamps')
        if len(set(len(vals)
            for vals in kwarg.values()
            if isinstance(vals, Sequence)
           )) > 1:
            raise ValueError('All time series must have same length')


    def _prepare_value_forcing(self):
        """
        Format forcing from time series data
        to the form {timestamp: {parameter: value}}.

        Note
        ----
            dictionary lookup is significantly slower than list indexing, but
            obviates need to carry details of anticipated model run
            in state or forcing instance.
        """
        timestamps = self._values_as_series.pop('time')
        dict_of_dicts = {}

        for i, time in enumerate(timestamps):
            dict_of_dicts[time] = {
                key: values[i]
                for key, values in self._values_as_series.items()
            }

        self._values = dict_of_dicts


class Forcing(Timeseries):
    """
    Time series forcing for model runs.
    """
    def force(self, state):
        """
        Force values of selected state parameters.
        """
        state.param.set_param(**self[state.time])

        for key, fn in self._callables.items():
            setattr(state, key, fn(state))
