"""W02 PR-3: the GitHub structured connector's LIVE wiring, now closed.

The connector drives a REAL W01 page walk (the operation is invoked, authorized
per page, and the walk advances on W01's single next_cursor) and reads the
fetched rows off ExecutionOutcome.payload (a CollectionPayload) — the neutral
data channel W01 added at RESULT_SCHEMA_VERSION=3 / EXECUTOR_SCHEMA_VERSION=4 /
PRODUCTION_SCHEMA_VERSION=3. fetch converts the rows with PR-2's converters,
folds them through diff_rows, and returns real (text, metadata); detect_changes
reports changed iff the since-window returned rows.

These tests use a real, isolated encrypted SecretVault and a self-signed HTTPS
loopback so the walk really talks TLS through the unmodified urllib_http_send:

* fetch returns real rows across >=2 pages, keyed and rendered, with the
  vault-resolved credential on the wire;
* detect_changes is True when the window returns rows;
* no provider still refuses NotImplementedError (PR-2's parked contract);
* a provider with no binding walks nothing and returns an empty-but-valid
  dataset, never a fabricated one.

No payload copy, no second envelope, no out-of-band capture, no global cache —
the rows come off the single ExecutionOutcome.payload and the cursor off the
single next_cursor.
"""

from __future__ import annotations

import asyncio
import datetime
import http.server
import ipaddress
import json
import ssl
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from kiro_crew.connections.control_plane.auth_modes import declare_permitted_modes
from kiro_crew.connections.control_plane.binding import binding_secret_ref, create_binding
from kiro_crew.connections.control_plane.handle import derive_handle, ensure_usable
from kiro_crew.connections.control_plane.policy import LayerCeilings
from kiro_crew.connections.control_plane.production import (
    BindingSecretSelector,
    urllib_http_send,
)
from kiro_crew.connections.vendors.github.dispatch import build_github_transport
from kiro_crew.knowledge.connectors.github_structured import (
    ENTITY_COMMIT,
    ENTITY_PULL_REQUEST,
    GithubStructuredConnector,
    GithubTransport,
    LiveFetchError,
)
from kiro_crew.secrets import SecretVault

_T0 = 1_000_000.0
_GRANTED: Tuple[str, ...] = ("repo",)


def _verifier(*, claimed_subject, claimed_tenant, service_id):
    return {"subject_ref": "s", "tenant_ref": "t"}


def _bundle_for(vault: SecretVault, *, clock=lambda: _T0) -> GithubTransport:
    binding = create_binding(
        service_id="github", claimed_subject="octocat", claimed_tenant="acme",
        credential_mode="oauth_user", verifier=_verifier, slug="github",
    )
    handle = derive_handle(
        binding, granted_scopes=_GRANTED, requested_scopes=("repo",),
        now=_T0, ttl_seconds=3600.0,
    )
    view = ensure_usable(handle, now=_T0)
    selector = BindingSecretSelector(
        slug="github", binding_fingerprint=view.binding_fingerprint,
        service_id=view.service_id, credential_mode=view.credential_mode,
    )
    transport = build_github_transport(
        operation_id="gh_list_pull_requests", selector=selector,
        vault=vault, http_send=urllib_http_send,
    )
    return GithubTransport(
        transport=transport, handle=handle, offered_mode="oauth_user",
        permitted=declare_permitted_modes(("oauth_user",)),
        layers=LayerCeilings(), governance_scope="tools",
        governance_item="pulls.list", clock=clock,
    )


@pytest.fixture
def real_vault(tmp_path: Path) -> SecretVault:
    vault = SecretVault(tmp_path / "crewhome")
    vault.set_sync(binding_secret_ref("github")["name"], "gh-installation-token")
    return vault


# ── self-signed HTTPS loopback that pages ───────────────────────────────────
def _tls_material(tmp_path: Path) -> Tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cf = tmp_path / "cert.pem"
    kf = tmp_path / "key.pem"
    cf.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    kf.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    return cf, kf


class _Recorder:
    def __init__(self) -> None:
        self.requests: List[Dict[str, str]] = []
        self.paths: List[str] = []


def _paging_handler(recorder: _Recorder, port_ref: Dict[str, int]):
    class _H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            recorder.requests.append({k: v for k, v in self.headers.items()})
            recorder.paths.append(self.path)
            port = port_ref["port"]
            if self.path.startswith("/page2"):
                body = json.dumps([{
                    "number": 2, "title": "b", "state": "open",
                    "user": {"login": "octocat"}, "pull_request": {},
                    "updated_at": "2026-09-02T00:00:00Z",
                    "html_url": "https://github.com/octo/hello/pull/2",
                    "url": "https://api.github.com/repos/octo/hello/pulls/2",
                }]).encode()
                headers = {"Content-Type": "application/json"}
            else:
                body = json.dumps([{
                    "number": 1, "title": "a", "state": "open",
                    "user": {"login": "octocat"}, "pull_request": {},
                    "updated_at": "2026-09-01T00:00:00Z",
                    "html_url": "https://github.com/octo/hello/pull/1",
                    "url": "https://api.github.com/repos/octo/hello/pulls/1",
                }]).encode()
                headers = {
                    "Content-Type": "application/json",
                    "Link": f'<https://localhost:{port}/page2>; rel="next"',
                }
            self.send_response(200)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            return

    return _H


@contextmanager
def _https_server(handler_cls: Any, certfile: Path, keyfile: Path) -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(certfile), str(keyfile))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ── the tests ───────────────────────────────────────────────────────────────
def test_fetch_returns_real_rows_across_two_pages_over_tls(
    tmp_path: Path, real_vault: SecretVault, monkeypatch: pytest.MonkeyPatch,
) -> None:
    certfile, keyfile = _tls_material(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(certfile))
    rec = _Recorder()
    port_ref: Dict[str, int] = {"port": 0}
    with _https_server(_paging_handler(rec, port_ref), certfile, keyfile) as port:
        port_ref["port"] = port
        import kiro_crew.connections.vendors.github.locator as gh_locator
        monkeypatch.setattr(gh_locator, "GITHUB_API_BASE", f"https://localhost:{port}")
        connector = GithubStructuredConnector(
            transport_provider=lambda s, e: _bundle_for(real_vault) if e == ENTITY_PULL_REQUEST else None,
        )
        text, metadata = asyncio.run(
            connector.fetch({"id": "src-1", "repo_full_name": "octo/hello"}))
    # Real rows came back through ExecutionOutcome.payload across BOTH pages.
    assert metadata["row_count"] == 2
    assert metadata["repo_full_name"] == "octo/hello"
    assert len(metadata["primary_keys"]) == 2
    # Both PRs (numbers 1 and 2) are present, keyed and rendered.
    assert "github_pull_request" in text
    keys = metadata["primary_keys"]
    assert any(k.endswith("1") for k in keys)
    assert any(k.endswith("2") for k in keys)
    # The walk really turned two pages over TLS: page 1 template, page 2 cursor.
    assert len(rec.paths) == 2
    assert rec.paths[0].startswith("/repos/octo/hello/pulls")
    assert rec.paths[1].startswith("/page2")
    # Custody ran: the vault-resolved credential reached the wire.
    auths = [next((v for k, v in r.items() if k.lower() == "authorization"), None) for r in rec.requests]
    assert auths == ["Bearer gh-installation-token", "Bearer gh-installation-token"]


def test_detect_changes_true_when_since_window_returns_rows(
    tmp_path: Path, real_vault: SecretVault, monkeypatch: pytest.MonkeyPatch,
) -> None:
    certfile, keyfile = _tls_material(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(certfile))
    rec = _Recorder()
    port_ref: Dict[str, int] = {"port": 0}
    with _https_server(_paging_handler(rec, port_ref), certfile, keyfile) as port:
        port_ref["port"] = port
        import kiro_crew.connections.vendors.github.locator as gh_locator
        monkeypatch.setattr(gh_locator, "GITHUB_API_BASE", f"https://localhost:{port}")
        connector = GithubStructuredConnector(transport_provider=lambda s, e: _bundle_for(real_vault))
        changed = asyncio.run(connector.detect_changes({"id": "s", "repo_full_name": "octo/hello"}))
    assert changed is True  # the window returned rows -> changed
    assert len(rec.paths) >= 1


def test_provider_returning_none_for_every_kind_fetches_nothing(
    real_vault: SecretVault,
) -> None:
    # A provider with no binding for any kind walks nothing; the fetch returns an
    # empty-but-valid dataset (no rows), never a fabricated one.
    connector = GithubStructuredConnector(transport_provider=lambda s, e: None)
    text, metadata = asyncio.run(connector.fetch({"id": "s", "repo_full_name": "o/r"}))
    assert metadata["row_count"] == 0
    assert text == ""


def test_no_transport_still_refuses_notimplemented() -> None:
    # PR-2's parked contract: no provider -> NotImplementedError, never a mock.
    connector = GithubStructuredConnector()
    with pytest.raises(NotImplementedError):
        asyncio.run(connector.fetch({"id": "s", "repo_full_name": "o/r"}))
    with pytest.raises(NotImplementedError):
        asyncio.run(connector.detect_changes({"id": "s", "repo_full_name": "o/r"}))


# ── per-row ingest contract (real SourceRow API, integrated) ────────────────
def _entity_routing_handler(recorder: _Recorder):
    """Route by path: issues/pulls/commits each return one item; a commit's
    check-runs return one check_run wrapped under `check_runs`."""

    class _H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            recorder.requests.append({k: v for k, v in self.headers.items()})
            recorder.paths.append(self.path)
            p = self.path
            if "/check-runs" in p:
                body = json.dumps({"check_runs": [{
                    "id": 555, "name": "ci", "head_sha": "abc123", "status": "completed",
                    "conclusion": "success", "completed_at": "2026-09-03T00:00:00Z",
                    "html_url": "h", "url": "https://api.github.com/repos/octo/hello/check-runs/555",
                }]}).encode()
            elif "/commits" in p:
                body = json.dumps([{
                    "sha": "abc123",
                    "commit": {"message": "m", "committer": {"name": "c", "date": "2026-09-02T00:00:00Z"}},
                    "html_url": "h", "url": "https://api.github.com/repos/octo/hello/commits/abc123",
                    "parents": [],
                }]).encode()
            elif "/issues" in p:
                body = json.dumps([{
                    "number": 9, "title": "iss", "state": "open", "user": {"login": "u"},
                    "updated_at": "2026-09-01T00:00:00Z",
                    "html_url": "h", "url": "https://api.github.com/repos/octo/hello/issues/9",
                }]).encode()
            else:  # pulls
                body = json.dumps([{
                    "number": 3, "title": "pr", "state": "open", "user": {"login": "u"},
                    "pull_request": {}, "updated_at": "2026-09-01T00:00:00Z",
                    "html_url": "h", "url": "https://api.github.com/repos/octo/hello/pulls/3",
                }]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            return

    return _H


def test_fetch_rows_covers_all_four_entities_incl_checkrun_fanout(
    tmp_path: Path, real_vault: SecretVault, monkeypatch: pytest.MonkeyPatch,
) -> None:
    certfile, keyfile = _tls_material(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(certfile))
    rec = _Recorder()
    with _https_server(_entity_routing_handler(rec), certfile, keyfile) as port:
        import kiro_crew.connections.vendors.github.locator as gh_locator
        monkeypatch.setattr(gh_locator, "GITHUB_API_BASE", f"https://localhost:{port}")
        connector = GithubStructuredConnector(
            transport_provider=lambda s, e: _bundle_for(real_vault))
        rows, snapshot, checkpoint = asyncio.run(
            connector.fetch_rows({"id": "src-1", "repo_full_name": "octo/hello"}))
    # issue + PR + commit + a check-run fanned out from that commit = 4 rows.
    assert len(rows) == 4
    assert {r.resource_ref.provider for r in rows} == {"github"}
    # The check-run row was fanned out per the commit sha (its ref path was hit).
    assert any("/check-runs" in p for p in rec.paths)
    assert any("/commits" in p for p in rec.paths)
    assert any("/issues" in p for p in rec.paths)
    assert any("/pulls" in p for p in rec.paths)
    assert any(r.resource_ref.locator.get("check_run_id") for r in rows)
    # Still fail-closed + incremental.
    assert all(r.subjects == () and r.tenant == "octo" for r in rows)
    assert snapshot is False


def test_supports_rows_is_true_with_the_real_ingest_api() -> None:
    # The real per-row ingest API (kiro_crew.knowledge.rows / acl) is integrated
    # into this branch, so the connector emits structured rows.
    from kiro_crew.knowledge.rows import SourceRow  # real symbol, not a stand-in
    assert SourceRow is not None
    assert GithubStructuredConnector().supports_rows() is True


def test_fetch_rows_returns_failclosed_sourcerows_over_tls(
    tmp_path: Path, real_vault: SecretVault, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercises the GENUINE SourceRow / ProviderResourceRef symbols (integrated),
    # not a stand-in.
    from kiro_crew.knowledge.acl import ProviderResourceRef
    from kiro_crew.knowledge.rows import SourceRow

    certfile, keyfile = _tls_material(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(certfile))
    rec = _Recorder()
    port_ref: Dict[str, int] = {"port": 0}
    with _https_server(_paging_handler(rec, port_ref), certfile, keyfile) as port:
        port_ref["port"] = port
        import kiro_crew.connections.vendors.github.locator as gh_locator
        monkeypatch.setattr(gh_locator, "GITHUB_API_BASE", f"https://localhost:{port}")
        connector = GithubStructuredConnector(
            transport_provider=lambda s, e: _bundle_for(real_vault) if e == ENTITY_PULL_REQUEST else None,
        )
        assert connector.supports_rows() is True
        rows, snapshot, checkpoint = asyncio.run(
            connector.fetch_rows({"id": "src-1", "repo_full_name": "octo/hello"}))
    # Real rows across BOTH pages, each a genuine SourceRow with fail-closed ACL.
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, SourceRow)
        assert row.subjects == ()            # fail-closed deny-all, never public
        assert row.tenant == "octo"          # vendor org/login, non-empty
        assert row.managed is True           # fixed by the DTO
        assert isinstance(row.resource_ref, ProviderResourceRef)
        assert row.resource_ref.provider == "github"
        assert row.resource_ref.locator["owner"] == "octo"
        assert row.resource_ref.locator["repo"] == "hello"
        assert "number" in row.resource_ref.locator  # a PR locator
    # Incremental (snapshot=False): absent rows are NOT deleted; checkpoint is
    # the advanced watermark from the fetched rows.
    assert snapshot is False
    assert checkpoint == "2026-09-02T00:00:00Z"  # max updated_at across pages
    # The walk really turned two pages over TLS.
    assert rec.paths[0].startswith("/repos/octo/hello/pulls")
    assert rec.paths[1].startswith("/page2")


def test_missing_transport_for_detect_changes_fails_closed() -> None:
    connector = GithubStructuredConnector(transport_provider=lambda s, e: None)
    with pytest.raises(LiveFetchError):
        asyncio.run(connector.detect_changes({"repo_full_name": "o/r"}))


def test_commit_and_pr_kinds_are_wired() -> None:
    from kiro_crew.knowledge.connectors.github_structured import _OP_FOR_ENTITY
    assert _OP_FOR_ENTITY[ENTITY_PULL_REQUEST] == "gh_list_pull_requests"
    assert _OP_FOR_ENTITY[ENTITY_COMMIT] == "gh_list_commits"
