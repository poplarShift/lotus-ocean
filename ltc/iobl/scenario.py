import param

from .state import State
from .model import Model

class Scenario(param.Parameterized):

    state = param.ClassSelector(State, allow_None=False, doc="""
        Physical variables determining the state of the turbulent oceanic boundary
        layer.""")

    model = param.ClassSelector(Model, allow_None=False, doc="""
        A model that calculates turbulence parameters for a given state.""")

    def steady_ltc(self):
        self.model.init_ekman_spiral(self)
        self.model.find_steady_k(self)
        self.state.update_ctd()
        self.model.update_interface(self)
