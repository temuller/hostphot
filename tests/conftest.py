"""Tests that need the external archives are marked with ``@pytest.mark.network`` (run them with
``pytest -m network``; the default CI runs ``pytest -m "not network"``). Network tests that fail because an
archive is unreachable or fails (connection errors, timeouts or server errors) are reported as skipped."""
import socket
import urllib.error

import pytest
import requests

NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
)
SERVER_ERROR_MESSAGES = ("500", "502", "503", "504", "Service Unavailable", "Gateway Time", "Unable to create connection")


def _is_network_error(exc: BaseException) -> bool:
    """Whether an exception (or one in its chain, e.g. re-raised by ``multi_band_phot``) is due to the archives."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, NETWORK_ERRORS):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            if response is not None and response.status_code >= 500:
                return True
        if any(message in str(exc) for message in SERVER_ERROR_MESSAGES) and "HTTP" in type(exc).__name__ + str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and item.get_closest_marker("network") is not None
        and call.excinfo is not None
        and _is_network_error(call.excinfo.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Archive not available: {call.excinfo.value!r}"[:300])
