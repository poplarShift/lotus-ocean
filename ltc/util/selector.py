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
        super().__getattribute__('select_active')()

    def select_active(self):
        """
        Set appropriate instance to active
        """
        fn = super().__getattribute__('_selector_fn')
        args = super().__getattribute__('_selector_args')
        # super().__setattr__('active', fn(**args))
        self.active = fn(**args)

    def __getattribute__(self, attr):
        """
        Wrap any attribute of the underlying instance.
        Activates appropriate instance on the fly.
        """
        super().__getattribute__('select_active')()
        active = super().__getattribute__('active')
        return getattr(active, attr)
