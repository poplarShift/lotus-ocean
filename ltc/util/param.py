from collections import namedtuple
import inspect
import param
import numpy as np

Log = namedtuple("event", "callstack param oldval newval")


class ValueSetLogger:
    """
    A context manager to log when parameter values change.
    """

    def __init__(self, *instances) -> None:
        self.instances = instances

    def __enter__(self):
        self._param_mod_events_backups = []
        for instance in self.instances:
            instance.param.debug("param logging - set _param_mod_events to []")
            self._param_mod_events_backups.append(instance._param_mod_events)
            instance._param_mod_events = []
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._param_mod_events_backups):
            instance._param_mod_events = ev_backup
            instance.param.debug("param logging - restore _param_mod_events")

    @property
    def events(self):
        events = []
        for instance in self.instances:
            events.extend(instance._param_mod_events)
        return events


class ParamForcer:
    """
    A context manager to temporarily override params_forced.
    """

    def __init__(self, *instances) -> None:
        self.instances = instances

    def __enter__(self):
        self._params_forced_backup = []
        for instance in self.instances:
            instance.param.debug("Enabling param forcing - start")
            self._params_forced_backup.append(instance.params_forced)
            instance.params_forced = []

    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._params_forced_backup):
            instance.params_forced = ev_backup
            instance.param.debug("Enabling param forcing - stop")


class ParamWatchdog(param.Parameterized):
    """
    Each time a parameter changes, echo old and new values
    """

    watch_param_set_events = param.Boolean(
        default=True,
        constant=True,
        doc="""
    Each time a parameter changes, log event and echo the new and old values at the DEBUG log
    level""",
    )

    params_forced = param.List(
        default=None,
        doc="""
        List of parameters that are forced by the user or ltc.run.forcing. Cannot be modified internally.
        """,
    )

    params_dynamic = param.List(
        default=[],
        doc="""
        List of parameters that can be dynamically updated.
        """,
    )

    def __init__(self, *argv, **params):
        super().__init__(*argv, **params)
        self._param_mod_events = []

        if self.watch_param_set_events:
            self.param.watch(
                self._watch_param_set_events,
                list(self.param.params().keys()),
                precedence=0,
            )

        if self.params_forced is None:
            self.params_forced = [
                p
                for p in params.keys()
                if p in self.param.params()
                and p
                not in ["params_forced", "params_dynamic", "watch_param_set_events"]
                and p not in self.params_dynamic
            ]
            self.param.debug(
                f"DEFAULT: Setting params_forced from None to {self.params_forced}"
            )

        if set(self.params_forced).intersection(self.params_dynamic):
            raise ValueError("params_forced and params_dynamic must not overlap")

    def _watch_param_set_events(self, *events):
        """
        Hook to perform when parameters are set.
        1. check if parameter is forced
        """
        for e in events:
            if e.name in self.params_forced:
                raise ValueError(f"Trying to set '{e.name}' but it is forced.")
            self.param.debug(f"{e.name:<10} changed from {e.old} to {e.new}")
            self._param_mod_events.append(
                (
                    Log([fi.function for fi in inspect.stack()], e.name, e.old, e.new)
                    # ,[fi for fi in inspect.stack()]
            ))
