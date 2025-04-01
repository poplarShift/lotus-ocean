from datetime import datetime
from numpy import datetime64
from pandas import Timestamp
import pandas as pd, numpy as np
from numbers import Number


class Time:
    """
    Represents either day numbers or a datetime.
    Can add a number of days.
    """

    _timestep_converters = {
        Number: lambda dt: dt,
        Timestamp: lambda dt: pd.Timedelta(days=dt),
    }

    _timestamp_converters = {
        Number: lambda t: float(t),
        str: lambda t: Timestamp(str),
        datetime: lambda t: Timestamp(t),
        datetime64: lambda t: Timestamp(t),
        Timestamp: lambda t: t,
    }

    def __init__(self, value):
        if isinstance(value, Time):
            self = value
        else:
            for val_type, timestamp_converter in self._timestamp_converters.items():
                if isinstance(value, val_type):
                    self.value = timestamp_converter(value)

    @property
    def value_as_float(self):
        """
        Return the value cast to either number of days or in microseconds.
        """
        return as_float(self.value)

    def __add__(self, timestep):
        """
        Add hours to the timestamp.
        """
        for val_type, timestep_converter in self._timestep_converters.items():
            if isinstance(self.value, val_type):
                return Time(self.value + timestep_converter(timestep))

    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def __le__(self, other):
        return self.value.__le__(other.value)

    def __repr__(self):
        return "{val}".format(val=str(self.value))


def as_float(x):
    """
    Coerces a given value to a float or an array of floats.
    - Floats stay floats
    - datetimes become microseconds since the numpy.datetime64 epoch.
    """
    if isinstance(x, (float, int)):
        return x
    elif isinstance(x, (datetime, datetime64, Timestamp)):
        return np.datetime64(x).astype("datetime64[us]").astype(float)
    elif isinstance(x, pd.DatetimeIndex):
        return x.values.astype("datetime64[us]")
    elif isinstance(x, np.ndarray):
        if x.dtype.name.startswith("float") or x.dtype.name.startswith("int"):
            return x
        elif x.dtype.name.startswith("datetime64"):
            return x.astype("datetime64[us]").astype(float)

    raise ValueError("Not handled")
