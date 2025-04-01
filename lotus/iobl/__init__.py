"""
This module implements a number of classes that define an ice-ocean boundary layer
that predicts turbulent fluxes and uses them to be stepped forward in time.
"""

from .model import (
    SteadyLTCModel,
    IceOceanInterface,
)  # calculates turbulent fluxes based on LTC
