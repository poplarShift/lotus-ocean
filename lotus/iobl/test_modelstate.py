import pytest
from lotus.iobl.model import ModelState
from lotus.util.param import ValueSetLogger


def test_init():
    with pytest.raises(Exception):
        # missing latitude
        model = ModelState(
            time=0,
        )

    model = ModelState(
        time=0,
        latitude=70,
    )
