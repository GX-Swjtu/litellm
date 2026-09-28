"""
Regression tests for the pass-through endpoint auth-default fix
(GHSA-7h34-mmrh-6g58).

Two failures the fix closes:

1. ``PassThroughGenericEndpoint.auth`` defaulted to ``False`` — an
   admin who added a pass-through to ``general_settings`` without
   explicitly setting ``auth: true`` shipped an unauthenticated
   forwarder.
2. Setting ``auth: true`` was rejected at startup unless the operator
   had a LiteLLM Enterprise license, leaving OSS deployments with no
   safe configuration.

The fix flips the default to ``True`` (safe-by-default) and removes
the enterprise gate so OSS operators can register an authenticated
pass-through. The runtime check in ``user_api_key_auth.py`` also now
defaults to ``True`` so a config dict (raw, not Pydantic) without an
``auth`` key still requires authentication.
"""

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI, Request

from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.proxy.auth.user_api_key_auth import (
    check_api_key_for_custom_headers_or_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _register_pass_through_endpoint,
)


def test_passthrough_auth_defaults_to_true():
    # Regression: an admin who configures a pass-through without setting
    # auth explicitly used to ship an unauthenticated forwarder. The
    # default is now safe.
    endpoint = PassThroughGenericEndpoint(
        path="/canary-forwarder",
        target="https://postman-echo.com/get",
    )
    assert endpoint.auth is True


def test_passthrough_auth_can_still_be_explicitly_disabled():
    # Operators who genuinely need an unauthenticated forwarder (e.g.
    # public webhook receiver) can opt in explicitly.
    endpoint = PassThroughGenericEndpoint(
        path="/public-webhook",
        target="https://example.com/webhook",
        auth=False,
    )
    assert endpoint.auth is False


@pytest.mark.asyncio
async def test_register_passthrough_with_auth_true_works_for_oss(monkeypatch):
    # Regression: setting ``auth: true`` used to raise at startup
    # unless ``premium_user`` was True, leaving OSS with no safe
    # configuration.
    app = MagicMock(spec=FastAPI)
    visited: set = set()

    endpoint = PassThroughGenericEndpoint(
        path="/forwarder",
        target="https://example.com",
        auth=True,
    )

    # Should not raise; OSS premium_user=False is allowed to use auth=True.
    await _register_pass_through_endpoint(
        endpoint=endpoint,
        app=app,
        premium_user=False,
        visited_endpoints=visited,
    )


@pytest.mark.asyncio
async def test_runtime_check_treats_missing_auth_key_as_authenticated():
    # The runtime dispatch in user_api_key_auth pulls
    # pass_through_endpoints from general_settings as raw dicts (the
    # Pydantic default never applies). A dict without an ``auth`` key
    # must default to "authenticated" — without this, the previous
    # behaviour (``endpoint.get("auth") is not True`` -> True -> empty
    # auth) ships an unauthenticated forwarder.
    request = MagicMock()
    request.headers = {}
    raw_endpoint_no_auth_key = {
        "path": "/forwarder",
        "target": "https://example.com",
        # ``auth`` deliberately omitted
    }

    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/forwarder",
        pass_through_endpoints=[raw_endpoint_no_auth_key],
        api_key="sk-1234",
    )

    # Result is the api_key string (auth is REQUIRED for this endpoint
    # — flow continues to normal key validation), NOT an empty
    # ``UserAPIKeyAuth()`` (which was the unauthenticated-forwarder
    # bug).
    assert result == "sk-1234"


@pytest.mark.asyncio
async def test_runtime_check_explicit_auth_false_still_skips_validation(monkeypatch):
    # Operators who explicitly set ``auth: False`` get the legacy
    # behaviour — an empty UserAPIKeyAuth, no key required.
    from litellm.proxy._types import UserAPIKeyAuth

    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as passthrough

    monkeypatch.setattr(passthrough, "_registered_pass_through_routes", {})
    app = FastAPI()
    await _register_pass_through_endpoint(raw_endpoint_auth_false, app, False, set())
    request = Request(
        {
            "type": "http",
            "path": "/public-webhook",
            "method": "POST",
            "headers": [],
            "endpoint": app.routes[-1].endpoint,
        }
    )
    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/public-webhook",
        pass_through_endpoints=[raw_endpoint_auth_false],
        api_key="",
    )

    assert isinstance(result, UserAPIKeyAuth)


@pytest_asyncio.fixture
async def passthrough_gateway(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[FastAPI, list[httpx.Request]]]:
    """Real routing/auth/forwarding; only the upstream HTTP transport is replaced."""
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as passthrough
    import litellm.proxy.proxy_server as proxy
    from litellm.proxy._types import LiteLLMRoutes, ProxyException
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    monkeypatch.setattr(passthrough, "_registered_pass_through_routes", {})
    monkeypatch.setattr(LiteLLMRoutes.openai_routes, "_value_", list(LiteLLMRoutes.openai_routes.value))
    monkeypatch.setattr(proxy, "master_key", "sk-gateway-test-master")
    monkeypatch.setattr(proxy, "general_settings", {})
    monkeypatch.setattr(proxy, "user_custom_auth", None)
    monkeypatch.setattr(proxy, "prisma_client", None)
    app = FastAPI()
    app.add_exception_handler(ProxyException, proxy.openai_exception_handler)

    @app.get("/paddleocr/admin", dependencies=[Depends(user_api_key_auth)])
    @app.get("/paddleocr-neighbor", dependencies=[Depends(user_api_key_auth)])
    async def protected_builtin() -> dict:
        return {"builtin": True}

    requests: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        await request.aread()
        requests.append(request)
        return httpx.Response(200, json={"jobId": "test-job", "state": "done"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(
            passthrough, "get_async_httpx_client", lambda **kwargs: SimpleNamespace(client=upstream_client)
        )
        yield app, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("root_path", ["", "/gateway"])
async def test_submit_and_poll_forward_client_credentials(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
    root_path: str,
) -> None:
    app, requests = passthrough_gateway
    await _register_pass_through_endpoint(
        PassThroughGenericEndpoint(
            path="/paddleocr",
            target="https://ocr.example.com",
            auth=False,
            include_subpath=True,
            forward_headers=True,
        ),
        app,
        False,
        set(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, root_path=root_path),
        base_url="http://gateway.test" + root_path,
    ) as client:
        headers = {"Authorization": "Bearer upstream-test-token", "X-Client-Trace": "test-trace"}
        submitted = await client.post(
            "/paddleocr/api/v2/ocr/jobs",
            headers=headers,
            files={"file": ("sample.pdf", b"synthetic-pdf-content", "application/pdf")},
            data={"model": "PaddleOCR-VL-1.6"},
        )
        assert submitted.status_code == 200, submitted.text
        polled = await client.get("/paddleocr/api/v2/ocr/jobs/test-job?detail=true", headers=headers)
        assert polled.status_code == 200, polled.text
    assert [str(request.url) for request in requests] == [
        "https://ocr.example.com/api/v2/ocr/jobs",
        "https://ocr.example.com/api/v2/ocr/jobs/test-job?detail=true",
    ]
    assert all(request.headers["authorization"] == headers["Authorization"] for request in requests)
    assert all(request.headers["x-client-trace"] == "test-trace" for request in requests)
    assert b"synthetic-pdf-content" in requests[0].content
    assert b"PaddleOCR-VL-1.6" in requests[0].content
    assert requests[0].headers["content-type"].split("boundary=")[1].encode() in requests[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forward_headers,configured,expected",
    [
        (False, {}, None),
        (True, {}, "Bearer upstream-test-token"),
        (True, {"Authorization": "Bearer configured-token"}, "Bearer configured-token"),
    ],
)
async def test_header_forwarding_is_opt_in_and_configured_headers_win(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
    forward_headers: bool,
    configured: dict[str, str],
    expected: str | None,
) -> None:
    app, requests = passthrough_gateway
    await _register_pass_through_endpoint(
        PassThroughGenericEndpoint(
            path="/paddleocr",
            target="https://ocr.example.com",
            auth=False,
            forward_headers=forward_headers,
            headers=configured,
        ),
        app,
        False,
        set(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response = await client.get("/paddleocr", headers={"Authorization": "Bearer upstream-test-token"})
    assert response.status_code == 200, response.text
    assert requests[0].headers.get("authorization") == expected


@pytest.mark.asyncio
async def test_overlapping_routes_use_the_selected_target_and_auth(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
) -> None:
    app, requests = passthrough_gateway
    for path, target, auth in [
        ("/paddleocr", "https://ocr.example.com", False),
        ("/paddleocr/api/v2/ocr/jobs", "https://jobs.example.com/api/v2/ocr/jobs", False),
        ("/paddleocr/private", "https://private.example.com", True),
    ]:
        await _register_pass_through_endpoint(
            PassThroughGenericEndpoint(path=path, target=target, auth=auth, include_subpath=True, forward_headers=True),
            app,
            False,
            set(),
        )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        for path in ["/paddleocr/api/v2/ocr/jobs", "/paddleocr/api/v2/ocr/jobs/test-job"]:
            response = await client.get(path, headers={"Authorization": "Bearer upstream-test-token"})
            assert response.status_code == 200, response.text
        for path in ["/paddleocr/private/job", "/paddleocr/admin", "/paddleocr-neighbor"]:
            response = await client.get(path)
            assert response.status_code == 401, response.text
    assert [str(request.url) for request in requests] == [
        "https://jobs.example.com/api/v2/ocr/jobs",
        "https://jobs.example.com/api/v2/ocr/jobs/test-job",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_config", [{}, {"auth": True}, {"auth": None}])
async def test_raw_config_requires_auth_unless_explicitly_disabled(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
    auth_config: dict[str, bool | None],
) -> None:
    app, requests = passthrough_gateway
    await _register_pass_through_endpoint(
        {"path": "/paddleocr", "target": "https://ocr.example.com", "include_subpath": True, **auth_config},
        app,
        False,
        set(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        for path in ["/paddleocr", "/paddleocr/jobs/id"]:
            response = await client.get(path)
            assert response.status_code == 401, response.text
    assert requests == []


@pytest.mark.asyncio
async def test_method_specific_auth_does_not_open_other_methods(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
) -> None:
    app, requests = passthrough_gateway
    for methods, auth in [(["GET"], False), (["POST"], True)]:
        await _register_pass_through_endpoint(
            PassThroughGenericEndpoint(
                path="/paddleocr", target="https://ocr.example.com", auth=auth, include_subpath=True, methods=methods
            ),
            app,
            False,
            set(),
        )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        assert (await client.get("/paddleocr/job")).status_code == 200
        assert (await client.post("/paddleocr/job", json={})).status_code == 401
        assert (await client.delete("/paddleocr/job")).status_code == 405
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("include_subpath", [False, True])
async def test_management_forward_headers_survives_create_edit_and_reload(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
    include_subpath: bool,
) -> None:
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as passthrough
    import litellm.proxy.proxy_server as proxy
    from litellm.proxy._types import ConfigFieldInfo, ConfigFieldUpdate, UserAPIKeyAuth

    app, requests = passthrough_gateway
    stored: list[dict] = []

    async def read_config(**kwargs: object) -> ConfigFieldInfo:
        return ConfigFieldInfo(field_name="pass_through_endpoints", field_value=stored.copy())

    async def write_config(data: ConfigFieldUpdate, **kwargs: object) -> None:
        stored[:] = data.field_value

    monkeypatch.setattr(proxy, "get_config_general_settings", read_config)
    monkeypatch.setattr(proxy, "update_config_general_settings", write_config)
    admin_request = Request({"type": "http", "app": app})
    created = await passthrough.create_pass_through_endpoints(
        data=PassThroughGenericEndpoint(
            path="/paddleocr",
            target="https://ocr.example.com",
            auth=False,
            include_subpath=include_subpath,
            forward_headers=True,
        ),
        request=admin_request,
        user_api_key_dict=UserAPIKeyAuth(),
    )
    endpoint_id = created.endpoints[0].id
    assert stored[0]["forward_headers"] is True
    route = "/paddleocr/jobs/id" if include_subpath else "/paddleocr"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:

        async def send(expected_header: str | None) -> None:
            response = await client.get(route, headers={"Authorization": "Bearer upstream-test-token"})
            assert response.status_code == 200, response.text
            assert requests[-1].headers.get("authorization") == expected_header

        await send("Bearer upstream-test-token")
        # An unrelated edit must not reset an omitted forwarding/auth field.
        await passthrough.update_pass_through_endpoints(
            endpoint_id=endpoint_id,
            data=PassThroughGenericEndpoint(path="/paddleocr", target="https://ocr.example.com", timeout=120),
            request=admin_request,
            user_api_key_dict=UserAPIKeyAuth(),
        )
        await send("Bearer upstream-test-token")
        for enabled in [False, True]:
            await passthrough.update_pass_through_endpoints(
                endpoint_id=endpoint_id,
                data=PassThroughGenericEndpoint(
                    path="/paddleocr", target="https://ocr.example.com", forward_headers=enabled
                ),
                request=admin_request,
                user_api_key_dict=UserAPIKeyAuth(),
            )
            assert stored[0]["forward_headers"] is enabled
            await send("Bearer upstream-test-token" if enabled else None)
        restored = await passthrough._get_pass_through_endpoints_from_db(endpoint_id=endpoint_id)
        assert restored[0].forward_headers is True
        passthrough._registered_pass_through_routes.clear()
        await _register_pass_through_endpoint(restored[0], app, False, set())
        await send("Bearer upstream-test-token")


@pytest.mark.asyncio
async def test_upstream_jwt_is_not_validated_as_a_gateway_jwt(
    passthrough_gateway: tuple[FastAPI, list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy

    app, requests = passthrough_gateway
    monkeypatch.setattr(proxy, "general_settings", {"enable_jwt_auth": True})
    await _register_pass_through_endpoint(
        PassThroughGenericEndpoint(
            path="/paddleocr", target="https://ocr.example.com", auth=False, include_subpath=True, forward_headers=True
        ),
        app,
        False,
        set(),
    )
    upstream_jwt = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvY3IifQ.dummy-signature"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response = await client.post(
            "/paddleocr/jobs", headers={"Authorization": upstream_jwt}, json={"model": "upstream-ocr"}
        )
    assert response.status_code == 200, response.text
    assert requests[0].headers["authorization"] == upstream_jwt
    assert json.loads(requests[0].content)["model"] == "upstream-ocr"
