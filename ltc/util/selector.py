class InstanceSelector:
    """
    Dynamically select from a defined set of underlying model instances.

    Arguments
    ---------
        selector_fn: Returns underlying instance
        selector_args: dictionary of keyword arguments to pass into select_active
    """

    def __init__(self, selector_fn, selector_args):
        self._selector_fn = selector_fn
        self._selector_args = selector_args
        self.select_active()


    def select_active(self):
        """
        Set appropriate instance to active
        """
        fn = self._selector_fn
        args = self._selector_args
        self.active = fn(**args)


    def __getattr__(self, attr):
        """
        Wrap any attribute of the underlying instance.
        Activates appropriate instance on the fly.
        """
        if not hasattr(self.active, attr):
            raise AttributeError

        self.select_active()
        return getattr(self.active, attr)
