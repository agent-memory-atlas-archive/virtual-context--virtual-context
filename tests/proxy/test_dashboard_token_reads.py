"""With a dashboard token set, every dashboard data route requires it.

The token guarded only mutating routes, so request captures, export, events
and telemetry stayed readable without it.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from virtual_context.proxy.dashboard import register_dashboard_routes
from virtual_context.proxy.metrics import ProxyMetrics


@pytest.fixture
def client():
    app = FastAPI()
    register_dashboard_routes(app, ProxyMetrics(), state=None, dashboard_token="t0k")
    return TestClient(app)


@pytest.mark.regression("BUG-115")
@pytest.mark.parametrize("path", [
    "/dashboard/requests", "/dashboard/export", "/dashboard/telemetry",
    "/dashboard/settings", "/dashboard/conversations", "/dashboard/replay/status",
])
def test_reads_need_the_token(client, path):
    assert client.get(path).status_code == 403
    assert client.get(path, headers={"X-VC-Dashboard-Token": "t0k"}).status_code != 403
    assert client.get(path + "?token=t0k").status_code != 403


@pytest.mark.regression("BUG-115")
def test_the_page_itself_loads_without_the_token(client):
    assert client.get("/dashboard").status_code == 200


def test_without_a_token_the_dashboard_stays_open():
    app = FastAPI()
    register_dashboard_routes(app, ProxyMetrics(), state=None)
    assert TestClient(app).get("/dashboard/requests").status_code == 200
