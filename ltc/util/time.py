from numpy import datetime64
from pandas import Timedelta, Timestamp
from numbers import Number

class Time:
    """
    Represents either day numbers or a datetime.
    Can add a number of days.
    """

    _timestep_converters = {
        Number: lambda dt: dt,
        Timestamp: lambda dt: Timedelta(days=dt),
    }

    _timestamp_converters = {
        Number: lambda t: float(t),
        str: lambda t: Timestamp(str),
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

    def __add__(self, timestep):
        """
        Add hours to the timestamp.
        """
        for val_type, timestep_converter in self._timestep_converters.items():
            if isinstance(self.value, val_type):
                return Time(self.value + timestep_converter(timestep))

    def __lt__(self, other): return self.value.__lt__(other.value)
    def __le__(self, other): return self.value.__le__(other.value)

    def __repr__(self): return 'Time {val}'.format(val=repr(self.value))
