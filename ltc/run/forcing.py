"""
Time series of environmental forcing parameters.
"""
from collections.abc import Sequence
from copy import deepcopy
import param
import gsw
import numpy as np, pandas as pd

from ..util.time import Time

def _interpolate_by_index(df, on_values):
    """
    Interpolate a dataframe `df` on values `on_values` of its index.
    """

    return (
        df
        .merge(pd.DataFrame(None, pd.Index(on_values, name=df.index.name), dtype=float), left_index=True, right_index=True, how='outer', sort=True)
        .interpolate(method='index') # interpolate on index values
        .fillna(method='bfill') # pad uppermost values
        .reindex(on_values) # only retain the new indices
    )


class Timeseries(param.Parameterized):
    """
    Provide time series of observed values.

    Instantiate with either parameter-values or parameter-callable pairs
    """

    def __init__(self, time=None, **kwarg):
        self.callables = {}
        self.data = pd.DataFrame(index=time)

        self.update(**kwarg)

    def update(self, **kwarg):
        """
        """
        for key, value in kwarg.items():
            if callable(value):
                if key in self.data:
                    self.data.pop(key)
                self.callables[key] = value
            else:
                self.callables.pop(key, None)
                self.data[key] = value
        assert not set(self.callables) & set(self.data.columns)

    @classmethod
    def from_dataframe(cls, df):
        df = df.copy()
        if 'time' in df:
            time = df.pop('time')
        else:
            time = df.index
        new = cls(time=time)
        new.update(**df)
        return new

    def update_from_dataframe(self, df):
        if not self.data.index.equals(df.index):
            df = _interpolate_on_column(df, self.data.index)
            self.param.message('Interpolating new data to existing index')
        self.update(**df)

    def __getitem__(self, time):
        if isinstance(time, Time):
            time = time.value

        results = {}

        for key, callable_ in self.callables.items():
            results[key] = callable_(time)

        for key, series in self.data.items():
            if pd.Timestamp(time) in series.index:
                results[key] = series.loc[time]
            else:
                self.param.debug(f'Interpolating at time {time}')
                results[key] = np.interp(np.datetime64(time), series.index, series.values)

        return results

    def __getattr__(self, attr):
        if attr == 'time':
            return self.data.index
        elif attr in self.data:
            return self.data[attr]
        elif attr in self.callables:
            return self.callables[attr]
        else:
            raise AttributeError

class Forcing(Timeseries):
    """
    Time series forcing for model runs.
    """
    def force(self, state):
        """
        Force values of selected state parameters.
        """
        state.param.set_param(**self[state.time])

        for key, fn in self.callables.items():
            setattr(state, key, fn(state))
