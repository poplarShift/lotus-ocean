import pytest

from dashboard_complete import make_app

@pytest.fixture
def app():
    return make_app()

def test_startup(app):
    app