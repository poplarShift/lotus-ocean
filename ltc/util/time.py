import numpy as np
from numbers import Number

class Time:
    """
    Represents either day numbers or a datetime. Can add a number of days.
    """
    _timestep_converters = {
        Number: lambda dt: dt,
        np.datetime64: lambda dt: np.timedelta64(dt, 'D'),
    }

    def __init__(self, value):
        if isinstance(value, Time):
            self = value
        else:
            self.value = float(value) # enable comparison magic

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
