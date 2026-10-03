import pytest
from fastapi.testclient import TestClient

from workbench.api import create_app
from workbench.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, data_dir=tmp_path, mode="demo", api_token="", embedding_backend="bm25"
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client
