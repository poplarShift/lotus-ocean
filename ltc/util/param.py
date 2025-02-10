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


class ParamProtectFrozen:
    """
    A context manager to protect frozen parameters.

    Only useful for instances with protect_frozen == False.
    """

    def __init__(self, *instances) -> None:
        self.instances = instances

    def __enter__(self):
        self._params_frozen_backup = []
        for instance in self.instances:
            instance.param.debug("Protecting frozen params - start")
            instance.protect_frozen = False

    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._params_frozen_backup):
            instance.protect_frozen = True
            instance.param.debug("Protecting frozen params - stop")


class ParamFreezer:
    """
    A context manager to temporarily freeze selected params.
    """

    def __init__(self, *instances, freeze=None) -> None:
        self.instances = instances
        self.frozen = freeze

    def __enter__(self):
        self._params_dynamic_backup = []
        self._params_frozen_backup = []

        for instance in self.instances:
            instance.param.debug("Enabling param forcing - start")
            self._params_dynamic_backup.append(instance.params_dynamic)
            self._params_frozen_backup.append(instance.params_frozen)
            instance.params_dynamic = []
            instance.params_frozen += self.frozen

    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._params_dynamic_backup):
            for instance, ev_backup in zip(self.instances, self._params_frozen_backup):
                instance.params_dynamic = ev_backup
                instance.params_frozen = ev_backup

            instance.param.debug("Enabling param forcing - stop")


class ParamUnfreezer:
    """
    A context manager to temporarily override params_frozen.
    """

    def __init__(self, *instances) -> None:
        self.instances = instances

    def __enter__(self):
        self._params_frozen_backup = []
        for instance in self.instances:
            instance.param.debug("Enabling param forcing - start")
            self._params_frozen_backup.append(instance.params_frozen)
            instance.params_frozen = []

    def __exit__(self, exc_type, exc_value, exc_tb):
        for instance, ev_backup in zip(self.instances, self._params_frozen_backup):
            instance.params_frozen = ev_backup
            instance.param.debug("Enabling param forcing - stop")


class ParamWatchdog(param.Parameterized):
    """
    Each time a parameter changes, echo old and new values
    """

    watch_param_set_events = param.Boolean(
        default=True, constant=True, doc="""Each time a parameter changes, log event."""
    )

    watch_param_set_events_detail = param.Boolean(
        default=False, constant=True, doc="""Each time a parameter changes, log
        old values, new values, and call stack."""
    )

    protect_frozen = param.Boolean(
        default=True,
        constant=True,
        doc="""
    Disallow changing frozen parameters.
    """,
    )

    params_frozen = param.List(
        default=None,
        doc="""
        List of parameters that are frozen by the user or ltc.run.forcing. Cannot be modified internally
        if protect_frozen is True (default).
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

        if self.params_frozen is None:
            self.params_frozen = [
                p
                for p in params.keys()
                if p in self.param.params()
                and p
                not in [
                    "params_frozen",
                    "params_dynamic",
                    "watch_param_set_events",
                    "protect_frozen",
                ]
                and p not in self.params_dynamic
            ]
            self.param.debug(
                f"DEFAULT: Setting params_frozen from None to {self.params_frozen}"
            )

        if set(self.params_frozen).intersection(self.params_dynamic):
            raise ValueError("params_frozen and params_dynamic must not overlap")

    def _watch_param_set_events(self, *events):
        """
        Hook to perform when parameters are set.
        1. check if parameter is frozen
        2. log event
        """
        for e in events:
            if e.name in self.params_frozen and self.protect_frozen:
                raise ValueError(f"Trying to set '{e.name}' but it is frozen.")
            if self.watch_param_set_events_detail:
                self.param.debug(
                    f"{e.name:<10} changed from {e.old} to {e.new} in {[fi.function for fi in inspect.stack()]}"
                )
            else:
                self.param.debug(f"{e.name:<10} changed.")
            self._param_mod_events.append(
                (
                    Log([fi.function for fi in inspect.stack()], e.name, e.old, e.new)
                    # ,[fi for fi in inspect.stack()]
                )
            )
