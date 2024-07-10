from collections import namedtuple
import inspect
import param
import numpy as np

Log = namedtuple('event', 'callstack param oldval newval')

class ValueSetLogger():
    def __init__(self, *instances) -> None:
        self.instances = instances

    def __enter__(self):
        self._param_mod_events_backups = []
        for instance in self.instances:
            self._param_mod_events_backups.append(instance._param_mod_events)
            self._param_mod_events = []
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._param_mod_events_backups):
            instance._param_mod_events = ev_backup

    @property
    def events(self):
        events = []
        for instance in self.instances:
            events.extend(instance._param_mod_events)
        return events


class LogValueSet(param.Parameterized):
    """
    Each time a parameter changes, echo old and new values
    """

    log_value_set_events = param.Boolean(default=True, constant=True, doc="""
    Each time a parameter changes, log event and echo the new and old values at the DEBUG log
    level""")

    def __init__(self, **params):
        super().__init__(**params)
        self._param_mod_events = []
        if self.log_value_set_events:
            self.param.watch(
                self._log_value_set_events, list(self.param.params().keys()),
                precedence=0,
            )

    def _log_value_set_events(self, *events):
        """
        TODO: log this into custom tuple
        """
        for e in events:
            self.param.debug(f'{e.name:<10} changed from {e.old} to {e.new}')
            self._param_mod_events.append(Log([fi.function for fi in inspect.stack()], e.name, e.old, e.new))
