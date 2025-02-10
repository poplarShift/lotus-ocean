import pytest

from dashboard_simple import make_app

@pytest.fixture
def app():
    return make_app()

def test_startup(app):
    app