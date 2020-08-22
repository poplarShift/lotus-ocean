"""
This module implements a number of classes that define an ice-ocean boundary layer
that predicts turbulent fluxes and uses them to be stepped forward in time.
"""

from .state import IceOceanBoundaryLayer # contains state such as temp., sal., fluxes
from .model import SteadyLTC # calculates turbulent fluxes based on LTC
