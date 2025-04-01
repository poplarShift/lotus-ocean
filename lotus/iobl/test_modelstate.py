import pytest
from ltc.iobl.model import ModelState
from ltc.util.param import ValueSetLogger


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
