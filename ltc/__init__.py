"""
Model turbulent transfer in the oceanic boundary layer using a "local turbulence
closure" scheme where eddy viscosity scales as the product of an empirically
determined mixing length and a turbulent velocity scale.

"""

from .util.version import read_version
__version__ = read_version()

def link_components(**components):
    """
    Relying on the canonical naming conventions in the different classes,
    link instances of all model components together.
    """

    def _find_component_params(inst):
        """
        Return list of all parameters of an instance that represents
        a canonical component
        """
        _canonical_components = ['state', 'model', 'stepper', 'run', 'observations',
                                 'forcing']
        all_params = list(inst.params().keys())
        canonical_params = [p for p in all_params if p in _canonical_components]
        return canonical_params

    def _set_param_subset(inst, subset_list, **params):
        """
        Set a subset of params on an instance
        """
        subset_dict = {k: v for k, v in params.items() if k in subset_list}
        inst.set_param(**subset_dict)
        return subset_dict

    def _format_dict(dct):
        lines = [''] + ['{:<12}: {}'.format(k, v) for k, v in dct.items()]
        return '\n'.join(lines)

    for inst in components.values():
        subset_list = _find_component_params(inst)
        subset_dict = _set_param_subset(inst, subset_list, **components)
        subset_repr = _format_dict({k: v.__str__() for k, v in subset_dict.items()})
        msg = """Registered components: {}
        """.format(subset_repr)
        inst.param.message(msg)
