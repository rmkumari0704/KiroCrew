"""GitHub structured-source connector.

GitHub's issues, pull requests, commits and check-runs are not prose: each is a
record with a stable identity and typed fields. This connector turns those
records into the knowledge Library's existing row shape so they are searchable
and refreshable like any other source, WITHOUT a second knowledge base, registry
or sync engine — it is a :class:`BaseConnector`, driven by the same
``SyncScheduler`` as ``local_folder``.

What it owns, and what it borrows:

* **Owns** the domain model — the typed rows below, the primary keys that make a
  refresh idempotent, the conversion from a GitHub API payload into a row, the
  ``since`` + primary-key-diff that makes a refresh incremental, and the
  resumable checkpoint that lets a refresh interrupted mid-run continue instead
  of restarting.
* **Borrows** everything else. The checkpoint lives in the source's existing
  ``properties`` blob (the same per-source state the scheduler already reads for
  ``last_synced`` / ``consecutive_failures``); no new table is added. Ingestion,
  de-duplication, storage and per-row lineage recording stay in the pipeline the
  scheduler already calls.

**Live fetch is not wired here.** Reaching GitHub needs the real transport
executor from the connections stack, which is not on ``main`` yet, and
multi-page extraction consumes ``connections.vendors.github`` pagination once
that lands rather than growing a second copy of paging here. Until then
:meth:`fetch` refuses rather than returning a partial or mocked dataset: a mock
is not a live read and must never be presented as one. Everything else — the
typed rows, the keys, the conversion, the incremental diff, the checkpoint and
the per-row lineage — is exercised on ``main`` today by the pure functions this
module exposes and their tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol

from ..acl import ProviderResourceRef
from ..rows import SourceRow
from .base import BaseConnector

if TYPE_CHECKING:  # type-only; the pure PR-2 surface imports with no connections dep
    from kiro_crew.connections.control_plane.operation import CredentialMode
# ── entity types ───────────────────────────────────────────────────────────
# One string per GitHub record kind. Stored verbatim on every row's lineage so a
# search hit can say which kind of GitHub object it came from.
ENTITY_ISSUE = "github_issue"
ENTITY_PULL_REQUEST = "github_pull_request"
ENTITY_COMMIT = "github_commit"
ENTITY_CHECK_RUN = "github_check_run"

SOURCE_TYPE = "github"

# The default GitHub instance host. A GitHub Enterprise Server deployment is a
# DIFFERENT instance, so the same ``owner/repo/123`` on two hosts is two
# distinct records and must key apart. Callers pass their own instance; this is
# the public-github fallback.
DEFAULT_INSTANCE = "github.com"

# Primary-key segment separator. A vertical bar cannot occur in a GitHub
# instance host, repo full name, entity kind or numeric/sha entity key, so it
# joins the domain segments without ambiguity. Every segment is checked for it
# at construction time so a value carrying the separator can never forge a key.
_PK_SEP = "|"

# ``owner/name`` — the shape every GitHub repository full name takes. Anchored so
# a stray path segment or a trailing slash is rejected rather than silently
# accepted as a repo.
_REPO_FULL_NAME_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


class LineageError(ValueError):
    """A row was built without complete provenance.

    Raised by :meth:`RowLineage.validate`. Provenance is not optional: a row a
    search can surface without being able to say where it came from is worse
    than an absent row, so the conversion fails loudly rather than storing one.
    """


@dataclass(frozen=True)
class RowLineage:
    """Per-row provenance and the full primary-key domain for every GitHub row.

    Every typed row carries exactly one of these. It answers, for one stored
    row: which knowledge SOURCE it belongs to, which GitHub INSTANCE (host) it
    came from, which REPOSITORY within that instance, what KIND of object it is,
    the primary key that identifies it, the GitHub API URL it was read from, and
    when it was read.

    ``source_id`` + ``instance`` + ``repo_full_name`` + ``entity_type`` are the
    key DOMAIN — the qualifiers that keep two records that share an entity key
    (the same issue number, the same short sha suffix) from colliding when they
    belong to different sources, hosts, repos or kinds. The kind's own entity
    key (number / sha / id) is appended to that domain by
    :func:`primary_key_for`. Without the domain, one repo's issue #1 and another
    repo's issue #1 would collapse to one stored row.
    """

    source_id: str
    instance: str
    repo_full_name: str
    entity_type: str
    primary_key: str
    api_url: str
    fetched_at: str

    def validate(self) -> None:
        """Raise :class:`LineageError` unless every fact is present and well-formed."""
        # The four key-domain segments must all be present, and none may carry
        # the segment separator, or a value could forge a different key.
        for label, value in (
            ("source_id", self.source_id),
            ("instance", self.instance),
        ):
            if not (value or "").strip():
                raise LineageError(f"{label} is required")
            if _PK_SEP in value:
                raise LineageError(f"{label} may not contain {_PK_SEP!r}")
        if not _REPO_FULL_NAME_RE.match(self.repo_full_name or ""):
            raise LineageError(
                f"repo_full_name must be 'owner/name', got {self.repo_full_name!r}")
        if self.entity_type not in _ENTITY_TYPES:
            raise LineageError(f"unknown entity_type {self.entity_type!r}")
        if not (self.primary_key or "").strip():
            raise LineageError("primary_key is required")
        if not (self.api_url or "").strip():
            raise LineageError("api_url is required")
        if not (self.fetched_at or "").strip():
            raise LineageError("fetched_at is required")


_ENTITY_TYPES = frozenset(
    {ENTITY_ISSUE, ENTITY_PULL_REQUEST, ENTITY_COMMIT, ENTITY_CHECK_RUN})


# ── typed rows ─────────────────────────────────────────────────────────────
# Each row is a record with real fields, not an opaque blob. The class-level
# PRIMARY_KEY_FIELDS name the fields whose values form the row's identity; that
# identity — never the title — is the dedup/idempotency key across refreshes.

@dataclass(frozen=True)
class IssueOrPullRow:
    """A GitHub issue or pull request.

    The kind's entity key is ``number``: GitHub numbers issues and pull requests
    in one shared per-repository sequence, so the number is unique within a repo.
    The repository, instance and source are separate segments of the full key
    domain (see :class:`RowLineage`), so ``number`` alone is the entity part. The
    title is a mutable field, never part of the key — the same issue keeps its
    identity when it is renamed.
    """

    PRIMARY_KEY_FIELDS = ("number",)

    repo_full_name: str
    number: int
    is_pull_request: bool
    title: str
    state: str
    author: str | None
    body: str
    labels: tuple[str, ...]
    created_at: str
    updated_at: str
    closed_at: str | None
    html_url: str
    lineage: RowLineage

    @property
    def primary_key(self) -> str:
        return primary_key_for(self)


@dataclass(frozen=True)
class CommitRow:
    """A GitHub commit. Identity is the ``sha`` — globally unique by construction."""

    PRIMARY_KEY_FIELDS = ("sha",)

    repo_full_name: str
    sha: str
    message: str
    author_name: str | None
    author_email: str | None
    authored_date: str | None
    committer_name: str | None
    committed_date: str | None
    parents: tuple[str, ...]
    html_url: str
    lineage: RowLineage

    @property
    def primary_key(self) -> str:
        return primary_key_for(self)


@dataclass(frozen=True)
class CheckRunRow:
    """A GitHub check-run. Identity is the numeric check-run ``id``."""

    PRIMARY_KEY_FIELDS = ("id",)

    repo_full_name: str
    id: int
    name: str
    head_sha: str
    status: str
    conclusion: str | None
    started_at: str | None
    completed_at: str | None
    details_url: str | None
    html_url: str
    lineage: RowLineage

    @property
    def primary_key(self) -> str:
        return primary_key_for(self)


TypedRow = IssueOrPullRow | CommitRow | CheckRunRow


def primary_key_for(row: TypedRow) -> str:
    """The row's identity string over the FULL key domain.

    Composed as ``source_id | instance | repo_full_name | entity_type |
    <entity-key>``, where the entity key is the row's declared
    ``PRIMARY_KEY_FIELDS`` joined by ``/``. The four domain segments come from
    the row's lineage, so two records that share an entity key never collapse
    when they belong to different sources, GitHub hosts, repositories or kinds:
    one repo's issue #1 and another repo's issue #1 are distinct keys, and so
    are the same repo mirrored under two knowledge sources or two instances.

    A single stable string so refreshes can diff by identity with set
    operations. Each entity-key part is the field value rendered verbatim.
    """
    entity_key = "/".join(str(getattr(row, f)) for f in row.PRIMARY_KEY_FIELDS)
    lin = row.lineage
    return _PK_SEP.join(
        (lin.source_id, lin.instance, lin.repo_full_name, lin.entity_type, entity_key))


# ── conversion: GitHub payload shape → typed row ────────────────────────────
# Each takes a repository full name, one raw GitHub API object, and the read
# timestamp, and returns one typed row with validated lineage. A malformed
# payload (missing required identity field, bad repo name) raises rather than
# producing a row that cannot be keyed or traced.


def _require(payload: dict, key: str, entity: str):
    if key not in payload or payload[key] is None:
        raise LineageError(f"{entity} payload missing required field {key!r}")
    return payload[key]


def _labels(payload: dict) -> tuple[str, ...]:
    out: list[str] = []
    for lbl in payload.get("labels") or ():
        if isinstance(lbl, dict):
            name = lbl.get("name")
        else:
            name = lbl
        if name:
            out.append(str(name))
    return tuple(out)


def _make_lineage(
    *, source_id: str, instance: str, repo_full_name: str, entity_type: str,
    entity_key: str, api_url: str, fetched_at: str,
) -> RowLineage:
    """Build a validated lineage whose ``primary_key`` is the full-domain key.

    The composed key is ``source_id | instance | repo | kind | entity_key`` — the
    same string :func:`primary_key_for` derives from a finished row, computed
    here so the lineage a row carries and the row's ``primary_key`` property
    always agree.
    """
    primary_key = _PK_SEP.join(
        (source_id, instance, repo_full_name, entity_type, entity_key))
    lineage = RowLineage(
        source_id=source_id,
        instance=instance,
        repo_full_name=repo_full_name,
        entity_type=entity_type,
        primary_key=primary_key,
        api_url=api_url,
        fetched_at=fetched_at,
    )
    lineage.validate()
    return lineage


def issue_or_pull_from_payload(
    repo_full_name: str, payload: dict, *, source_id: str, fetched_at: str,
    instance: str = DEFAULT_INSTANCE,
) -> IssueOrPullRow:
    number = int(_require(payload, "number", "issue/pull"))
    # GitHub's issues API returns pull requests too, marked by a ``pull_request``
    # sub-object; the pulls API omits it but every PR object carries it. Either
    # signal makes this a PR.
    is_pr = "pull_request" in payload or payload.get("_is_pull_request", False)
    entity = ENTITY_PULL_REQUEST if is_pr else ENTITY_ISSUE
    user = payload.get("user") or {}
    lineage = _make_lineage(
        source_id=source_id, instance=instance, repo_full_name=repo_full_name,
        entity_type=entity, entity_key=str(number),
        api_url=str(_require(payload, "url", "issue/pull")), fetched_at=fetched_at)
    return IssueOrPullRow(
        repo_full_name=repo_full_name,
        number=number,
        is_pull_request=is_pr,
        title=str(payload.get("title") or ""),
        state=str(payload.get("state") or ""),
        author=(user.get("login") if isinstance(user, dict) else None),
        body=str(payload.get("body") or ""),
        labels=_labels(payload),
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        closed_at=(payload.get("closed_at") or None),
        html_url=str(payload.get("html_url") or ""),
        lineage=lineage,
    )


def commit_from_payload(
    repo_full_name: str, payload: dict, *, source_id: str, fetched_at: str,
    instance: str = DEFAULT_INSTANCE,
) -> CommitRow:
    sha = str(_require(payload, "sha", "commit"))
    commit = payload.get("commit") or {}
    author = commit.get("author") or {}
    committer = commit.get("committer") or {}
    parents = tuple(
        str(p.get("sha")) for p in (payload.get("parents") or ()) if p.get("sha"))
    lineage = _make_lineage(
        source_id=source_id, instance=instance, repo_full_name=repo_full_name,
        entity_type=ENTITY_COMMIT, entity_key=sha,
        api_url=str(_require(payload, "url", "commit")), fetched_at=fetched_at)
    return CommitRow(
        repo_full_name=repo_full_name,
        sha=sha,
        message=str(commit.get("message") or ""),
        author_name=(author.get("name") or None),
        author_email=(author.get("email") or None),
        authored_date=(author.get("date") or None),
        committer_name=(committer.get("name") or None),
        committed_date=(committer.get("date") or None),
        parents=parents,
        html_url=str(payload.get("html_url") or ""),
        lineage=lineage,
    )


def check_run_from_payload(
    repo_full_name: str, payload: dict, *, source_id: str, fetched_at: str,
    instance: str = DEFAULT_INSTANCE,
) -> CheckRunRow:
    run_id = int(_require(payload, "id", "check-run"))
    lineage = _make_lineage(
        source_id=source_id, instance=instance, repo_full_name=repo_full_name,
        entity_type=ENTITY_CHECK_RUN, entity_key=str(run_id),
        api_url=str(_require(payload, "url", "check-run")), fetched_at=fetched_at)
    return CheckRunRow(
        repo_full_name=repo_full_name,
        id=run_id,
        name=str(payload.get("name") or ""),
        head_sha=str(payload.get("head_sha") or ""),
        status=str(payload.get("status") or ""),
        conclusion=(payload.get("conclusion") or None),
        started_at=(payload.get("started_at") or None),
        completed_at=(payload.get("completed_at") or None),
        details_url=(payload.get("details_url") or None),
        html_url=str(payload.get("html_url") or ""),
        lineage=lineage,
    )


# ── incremental refresh: since + primary-key diff ───────────────────────────
# GitHub exposes no single delta token, so an incremental refresh is computed:
# ask only for records touched since the last watermark, then diff the returned
# keys against the keys already stored. The result names what to upsert and what
# has disappeared, and never relies on a title or ordinal position.


@dataclass(frozen=True)
class RefreshPlan:
    """The outcome of diffing a fetched key set against the stored key set."""

    upserts: tuple[TypedRow, ...]
    # Keys present in the store, absent from a FULL listing of the repo — i.e.
    # genuinely gone, safe for a caller to retire. Populated ONLY when the diff
    # was over a full listing (``full_listing=True``). On a windowed (``since``)
    # refresh this is always empty, because a window returns only CHANGED
    # records: "absent from this window" is the normal state of nearly every
    # stored key and says nothing about whether the record still exists.
    disappeared: tuple[str, ...]
    # The watermark to persist once this plan is applied — the max ``updated_at``
    # seen, so the next refresh asks only for what changed after it.
    next_since: str | None


def _max_timestamp(rows: tuple[TypedRow, ...]) -> str | None:
    stamps: list[str] = []
    for row in rows:
        stamp = getattr(row, "updated_at", None) or getattr(row, "committed_date", None) \
            or getattr(row, "completed_at", None) or row.lineage.fetched_at
        if stamp:
            stamps.append(stamp)
    return max(stamps) if stamps else None


def diff_rows(
    fetched: tuple[TypedRow, ...],
    stored_keys: frozenset[str],
    *,
    prior_since: str | None,
    full_listing: bool = False,
) -> RefreshPlan:
    """Diff a freshly-fetched set against what is already stored.

    ``fetched`` are the typed rows a query returned; ``stored_keys`` are the
    primary keys already in the source. Every fetched row is an upsert.

    ``full_listing`` says whether ``fetched`` is a COMPLETE listing of the repo
    (no ``since`` filter) or an incremental WINDOW. It decides ``disappeared``:

    * full listing -> a stored key not in ``fetched`` is genuinely gone, so it
      is reported for the caller to retire.
    * window (the default, incremental case) -> ``disappeared`` is always empty.
      A ``since`` window carries only CHANGED records, so nearly every stored
      key is "absent from this window" without being gone; reporting those would
      invite a caller to purge live records.

    A primary-key collision inside one fetched set (two rows sharing a key)
    keeps the LAST occurrence, matching an upsert's last-writer-wins semantics,
    so a page boundary that re-lists a straddling record is idempotent.
    """
    by_key: dict[str, TypedRow] = {}
    for row in fetched:
        by_key[row.primary_key] = row  # last-writer-wins on collision
    upserts = tuple(by_key.values())
    disappeared = (
        tuple(sorted(stored_keys - by_key.keys())) if full_listing else ())
    window_max = _max_timestamp(upserts)
    if window_max is None:
        next_since = prior_since
    elif prior_since is None:
        next_since = window_max
    else:
        # Never move the watermark backwards: a clock-skewed or re-listed record
        # with an older stamp must not reopen an already-covered window.
        next_since = max(window_max, prior_since)
    return RefreshPlan(upserts=upserts, disappeared=disappeared, next_since=next_since)


# ── resumable checkpoint (stored in the source's existing properties blob) ──
# No new table: the checkpoint is a small dict under a reserved key in the
# ``properties`` blob the scheduler already round-trips. It records the
# watermark and the position within a multi-entity, multi-page refresh so an
# interrupted run resumes instead of restarting.

_CHECKPOINT_KEY = "github_structured_checkpoint"

# The entity kinds a refresh walks, in a fixed order so a resume knows which
# kinds are already done and which remains.
REFRESH_ENTITY_ORDER = (ENTITY_ISSUE, ENTITY_COMMIT, ENTITY_CHECK_RUN)


@dataclass
class Checkpoint:
    """Where a refresh is up to, so it can resume after an interruption.

    ``since`` is the watermark for the *next* full refresh. ``entity_index`` and
    ``page_cursor`` locate the position inside an in-progress refresh: which
    entity kind (index into :data:`REFRESH_ENTITY_ORDER`) and which page cursor
    within it were last completed. ``in_progress`` distinguishes a clean
    watermark-only checkpoint from one paused mid-walk.
    """

    since: str | None = None
    entity_index: int = 0
    page_cursor: str | None = None
    in_progress: bool = False

    def to_dict(self) -> dict:
        return {
            "since": self.since,
            "entity_index": self.entity_index,
            "page_cursor": self.page_cursor,
            "in_progress": self.in_progress,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Checkpoint":
        if not data:
            return cls()
        try:
            entity_index = int(data.get("entity_index", 0))
        except (TypeError, ValueError):
            entity_index = 0
        # Clamp a persisted index that does not name a valid entity (a shorter
        # order between versions) so a resume starts over rather than indexing
        # out of range.
        if not 0 <= entity_index < len(REFRESH_ENTITY_ORDER):
            entity_index = 0
        return cls(
            since=(data.get("since") or None),
            entity_index=entity_index,
            page_cursor=(data.get("page_cursor") or None),
            in_progress=bool(data.get("in_progress", False)),
        )


def read_checkpoint(source: dict) -> Checkpoint:
    """Read the checkpoint out of a source's ``properties`` blob.

    ``source`` is a sources-row dict whose ``properties`` may already be a parsed
    dict (as the scheduler hands it) or a JSON string (as the raw row carries
    it); both are accepted so callers need not normalise first.
    """
    props = source.get("properties")
    if isinstance(props, str):
        try:
            props = json.loads(props or "{}")
        except ValueError:
            props = {}
    if not isinstance(props, dict):
        props = {}
    return Checkpoint.from_dict(props.get(_CHECKPOINT_KEY))


def write_checkpoint(properties: dict, checkpoint: Checkpoint) -> dict:
    """Return a copy of ``properties`` with the checkpoint written under its key.

    Pure: the caller persists the result through ``store.update_source`` (the
    existing per-source state write); nothing here touches the store, so the
    checkpoint logic is testable without a database.
    """
    updated = dict(properties or {})
    updated[_CHECKPOINT_KEY] = checkpoint.to_dict()
    return updated


# ── rendering a typed row into the pipeline's (text, metadata) shape ────────
# The scheduler-facing contract is prose-shaped: ``fetch`` returns one text blob
# plus metadata that the ingestion pipeline chunks and stores, and lineage is
# recorded per stored item. A typed row renders to a stable, human-readable text
# projection (so a search hit is legible) and a flat metadata dict carrying the
# primary key and the full lineage domain (so the stored item stays traceable).


def render_row_text(row: TypedRow) -> str:
    """A stable, legible text projection of a typed row for the search index."""
    d = asdict(row)
    d.pop("lineage", None)
    lines = [f"{row.lineage.entity_type} {row.primary_key}"]
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def render_row_metadata(row: TypedRow) -> dict:
    """A flat metadata dict: the primary key plus the full lineage domain."""
    return {
        "primary_key": row.primary_key,
        "source_id": row.lineage.source_id,
        "instance": row.lineage.instance,
        "repo_full_name": row.lineage.repo_full_name,
        "entity_type": row.lineage.entity_type,
        "api_url": row.lineage.api_url,
        "fetched_at": row.lineage.fetched_at,
    }


# ── PR-3: the live wiring (real invocation + paging through W01's transport) ─
# PR-2 above owns the domain model and refuses a live read until a real
# transport exists. This section drives that transport WITHOUT re-doing any of
# it: it opens W01's PageWalk (through connections.vendors.github.dispatch) and
# advances it page by page, so the operation is really invoked, authorized per
# page, and the per-page cursor really followed. The ONLY sender is W01's
# transport, composed by the caller and handed in.
#
# WHAT IS NOT CLOSED, AND WHY (root-confirmed at cd00f1837): the fetched page
# BODY does not come back. build_production_transport builds
# `TransportResponse(http_status=..., result=decode(reply))`, and a ResultDecode
# returns only an OperationResult ({status, next_cursor}); Decoded2xx does
# capture `body: bytes` but only `.result` propagates, so the rows are dropped
# before they reach ExecutionOutcome (which has no payload slot either). W01
# owns the fix (a single neutral payload slot threaded Decoded2xx.body ->
# TransportResponse -> ExecutionOutcome, with a schema bump). Until that
# versioned commit lands, the row-extraction step is left as an EXPLICIT,
# UNVERIFIED seam that FAILS CLOSED (raises) rather than fabricating rows or
# claiming a dataset. No workaround is used: no out-of-band body capture, no
# response cache, no local envelope re-declaration, no vendor side channel.

# The GitHub list operation per entity kind (operation_id only, never a URL, so
# auth/custody/paging all stay W01's). Only the entities with a REPO-SCOPED REST
# list op carrying {owner}/{repo} path params are wired here:
#   * pull requests -> gh_list_pull_requests (GET /repos/{owner}/{repo}/pulls),
#     converted by issue_or_pull_from_payload;
#   * commits       -> gh_list_commits       (GET /repos/{owner}/{repo}/commits),
#     converted by commit_from_payload.
# TWO entities are deliberately NOT wired, as open questions rather than
# hand-rolled URLs (a naked path is forbidden):
#   * issues     -- the repo-scoped issues LIST is not in the vendor table; the
#     only issue-listing ops are gh_search_issues (GET /search/issues, a `q=`
#     search shape, no {owner}/{repo} path) and gh_list_issues (GraphQL prose,
#     unshapeable by the REST locator);
#   * check-runs -- ENTITY_CHECK_RUN has NO list op at all (only a legacy
#     branch-protection PATCH).
# The repo-scoped list operation per entity kind (operation_id only, never a
# URL, so auth/custody/paging stay W01's). All three repo-scoped kinds are wired:
#   * issues       -> gh_list_issues_rest  (GET /repos/{owner}/{repo}/issues)
#   * pull requests -> gh_list_pull_requests (GET /repos/{owner}/{repo}/pulls)
#   * commits      -> gh_list_commits      (GET /repos/{owner}/{repo}/commits)
# CHECK-RUNS is the fourth entity but has NO repo-wide list: it hangs off a
# commit ref (GET /repos/{owner}/{repo}/commits/{ref}/check-runs, gh_list_check_runs),
# so it is walked per-commit as a dependent fan-out over the commits just fetched
# (see _walk_check_runs), not from this repo-scoped table.
_OP_FOR_ENTITY = {
    ENTITY_ISSUE: "gh_list_issues_rest",
    ENTITY_PULL_REQUEST: "gh_list_pull_requests",
    ENTITY_COMMIT: "gh_list_commits",
}

# The full entity->operation_id map the walk resolves against. It adds
# check-runs (walked per-commit-ref, not in the repo-scoped iteration set above)
# so _walk_entity_rows can resolve every kind's operation.
_OP_ALL = dict(_OP_FOR_ENTITY)
_OP_ALL[ENTITY_CHECK_RUN] = "gh_list_check_runs"

# Which PR-2 converter turns one raw GitHub payload item into a typed row, per
# wired entity. issue_or_pull_from_payload keys off the payload's pull_request
# sub-object, so it is correct for both the issues and pull-requests streams
# (the REST issues API returns PRs too, marked by that sub-object).
_CONVERTER_FOR_ENTITY = {
    ENTITY_ISSUE: issue_or_pull_from_payload,
    ENTITY_PULL_REQUEST: issue_or_pull_from_payload,
    ENTITY_COMMIT: commit_from_payload,
    ENTITY_CHECK_RUN: check_run_from_payload,
}


def _now_iso() -> str:
    """The read timestamp stamped onto every row's lineage (UTC, ISO 8601)."""

    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _source_id_of(source: dict) -> str:
    """The knowledge source id, from whichever key the row carries it under."""

    return str(source.get("id") or source.get("source_id") or "")


def _resource_ref_for(typed: Any, *, owner: str) -> ProviderResourceRef:
    """Build the GitHub :class:`ProviderResourceRef` locating one typed row.

    The locator shapes are the ones acl.ProviderResourceRef documents for GitHub:
    issue/pull ``{owner, repo, number}``, commit ``{owner, repo, sha}``. Provider
    is ``github``; account is the vendor org/login (the repo owner); resource_id
    is the row's own entity key. This is what the query-time gate resolves to
    revalidate, so a managed row cannot be stored without it.
    """

    repo_name = typed.repo_full_name.split("/", 1)[1]
    if isinstance(typed, IssueOrPullRow):
        locator: dict = {"owner": owner, "repo": repo_name, "number": typed.number}
        resource_id = str(typed.number)
    elif isinstance(typed, CommitRow):
        locator = {"owner": owner, "repo": repo_name, "sha": typed.sha}
        resource_id = typed.sha
    else:  # CheckRunRow and any future kind
        locator = {"owner": owner, "repo": repo_name, "check_run_id": getattr(typed, "id", "")}
        resource_id = str(getattr(typed, "id", ""))
    return ProviderResourceRef(
        provider="github", account=owner, resource_id=resource_id, locator=locator)


def _source_row_from(typed: Any, *, owner: str) -> SourceRow:
    """Map one PR-2 typed row to a :class:`SourceRow`, ACL fail-closed.

    ``key`` is the row's stable primary key; ``text`` is its own projection;
    ``resource_ref`` is the GitHub locator; ``tenant`` is the repo owner
    (non-empty). ``subjects`` is an EMPTY tuple -- explicit deny-all -- because
    this slice has NO authorization evidence mapping a GitHub object to the
    subjects allowed to see it, and a missing grant must never become public.
    ``managed`` is fixed True by the DTO. Real subjects/public need an authorized
    binding (repo visibility / collaborators), which is PR-4's live territory
    plus W01's binding identity -- not something this slice may synthesise.
    """

    return SourceRow(
        key=typed.primary_key,
        text=render_row_text(typed),
        subjects=(),  # fail-closed: no evidence -> deny-all, never public
        tenant=owner,  # the vendor org/login; non-empty
        resource_ref=_resource_ref_for(typed, owner=owner),
        title=typed.primary_key,
        item_type="document",
    )


@dataclass(frozen=True)
class GithubTransport:
    """Everything one entity's page walk needs, all resolved by W01 / the caller.

    The connector never builds a handle, selector, vault or transport itself --
    custody, auth and fencing are W01's, so the caller (the scheduler
    integration, a test) composes a real
    :func:`~kiro_crew.connections.vendors.github.dispatch.build_github_transport`
    plus the W01 handle/gate inputs and hands them in through a
    :class:`GithubTransportProvider`. ``clock`` is optional (a deterministic
    test injects one; production leaves it ``None`` so the executor reads its own
    server clock).
    """

    transport: Any
    handle: Any
    offered_mode: "CredentialMode"
    permitted: Any
    layers: Any
    governance_scope: str
    governance_item: str
    clock: Optional[Callable[[], float]] = None


class GithubTransportProvider(Protocol):
    """Resolves a source + entity to a :class:`GithubTransport`, or ``None``.

    The seam that keeps custody out of this module: given the source row and the
    entity kind about to be walked, an implementation returns the W01-composed
    transport bundle for that binding (per-binding via W01's
    ``BindingSecretSelector`` -- one binding, one transport), or ``None`` when it
    cannot. A ``None`` makes the connector fail closed for that entity rather
    than fabricate a read.
    """

    def __call__(
        self, source: dict, entity_type: str
    ) -> Optional["GithubTransport"]:  # pragma: no cover - Protocol
        ...


class LiveFetchError(RuntimeError):
    """A live GitHub read could not complete, so nothing is stored.

    Raised when a page walk's gate denied or the transport failed, when no W01
    transport is available for a source, or -- until W01's payload slot lands --
    when a walk turned pages but the row payload cannot be read back. It is a
    fail-closed refusal: a partial, empty, or page-count-only result is never
    presented as a complete live read. Distinct from :class:`NotImplementedError`,
    which is the "no transport configured at all" case PR-2's parked tests
    assert.
    """


def _repo_of(source: dict) -> str:
    """The ``owner/name`` this source reads, validated.

    Accepts the spellings ``validate_config`` accepts (``uri`` with an optional
    ``github://`` prefix) plus the ``repo_full_name`` the refresh tests pass.
    Refuses anything that is not ``owner/name`` -- a malformed repo must never be
    shaped into a request.
    """

    repo = (
        (source.get("repo_full_name") or source.get("uri") or "")
        .strip()
        .removeprefix("github://")
    )
    if not _REPO_FULL_NAME_RE.match(repo):
        raise LineageError(f"source repo must be 'owner/name', got {repo!r}")
    return repo


class GithubStructuredConnector(BaseConnector):
    """Consume GitHub issues/PRs/commits/check-runs as one structured source.

    Conforms to :class:`BaseConnector` so the existing ``SyncScheduler`` drives
    it. The typed-row conversion, primary-key diffing and checkpointing are the
    module-level pure functions above (PR-2); the live invocation + paging is
    wired through W01's executor/production transport (PR-3) via an injected
    :class:`GithubTransportProvider`.

    **Fail-closed without a transport.** Built with no ``transport_provider``
    (the default), :meth:`fetch` and :meth:`detect_changes` REFUSE with
    :class:`NotImplementedError` exactly as PR-2 shipped -- a mock read is not a
    live read.

    **Row payload is a pending W01 seam.** Even WITH a provider, the fetched page
    body does not yet come back through W01's ``ExecutionOutcome`` (root-confirmed
    at cd00f1837: only ``OperationResult`` = ``{status, next_cursor}`` propagates;
    ``Decoded2xx.body`` is dropped). W01 owns adding a neutral payload slot. Until
    that versioned commit lands, the live path drives a REAL page walk (the
    operation is invoked, authorized per page, and the per-page cursor followed)
    but then FAILS CLOSED at row extraction rather than fabricating rows or
    claiming a dataset. It re-implements no auth, custody, retry, fencing, error
    class or pagination.
    """

    def __init__(self, transport_provider: Optional[GithubTransportProvider] = None) -> None:
        self._transport_provider = transport_provider

    def source_type(self) -> str:
        return SOURCE_TYPE

    # ── per-row ingest contract (chat-408's SourceRow API), consumed in-slice ──
    # ── per-row ingest contract (SourceRow API), driven by the real scheduler ──
    def supports_rows(self) -> bool:
        """True: this connector emits structured ROWS (per-row ACL), not one text
        blob, so the SyncScheduler drives the per-row ingest path (fetch_rows)."""

        return True

    async def fetch_rows(self, source: dict):
        """Fetch the source's rows for the per-row ingest contract.

        Returns ``(rows, snapshot, checkpoint)``:

        * ``rows`` -- one ``kiro_crew.knowledge.rows.SourceRow`` per fetched
          record, each carrying its own ``key`` (the primary-key identity),
          ``text`` (the row's own projection), ``resource_ref`` (the GitHub
          :class:`ProviderResourceRef` locating this object), ``tenant`` (the
          vendor org/login, non-empty), and ``subjects``.
        * ``snapshot`` -- **False (incremental)**, chosen deliberately: this
          source refreshes by a ``since`` watermark, so a round returns only the
          records that changed after it. Absent rows are NOT gone — they simply
          did not change — so ``snapshot=True`` (which authorises DELETING absent
          rows) would destroy live rows every incremental round. A full-snapshot
          mode would need an unfiltered listing of every entity and is not what a
          ``since`` refresh produces.
        * ``checkpoint`` -- the opaque ``next_since`` watermark. The scheduler
          persists it at ``props['checkpoint']`` and advances it ONLY after the
          ingest reports every row fully persisted (RowsIngestOutcome
          .fully_persisted); a half-done batch leaves the old checkpoint, so the
          next round re-attempts the un-persisted rows.

        **ACL is fail-closed.** ``subjects`` is an EMPTY tuple (explicit
        deny-all) for every row: this slice has no authorization evidence that
        maps a GitHub object to the subjects allowed to see it, a MISSING grant
        must never become public, and real evidence needs an authorized binding
        (repo visibility / collaborators) that is PR-4's live territory plus
        W01's binding identity, not something this slice may synthesise. Deny-all
        is the confirmed-correct state until that evidence source exists.
        ``tenant`` is the repo owner (non-empty), ``managed`` is fixed True.
        """

        if self._transport_provider is None:
            raise NotImplementedError(
                "GitHub row fetch needs a transport provider that composes W01's "
                "executor/production; none was configured")

        repo = _repo_of(source)
        owner, name = repo.split("/", 1)
        checkpoint = read_checkpoint(source)
        rows: list = []
        commit_shas: list = []
        max_since = checkpoint.since
        for entity_type in _OP_FOR_ENTITY:
            bundle = self._transport_provider(source, entity_type)
            if bundle is None:
                continue
            base_args: dict = {"owner": owner, "repo": name}
            if checkpoint.since:
                base_args["since"] = checkpoint.since
            for typed in self._walk_entity_rows(
                entity_type=entity_type, repo=repo,
                source_id=_source_id_of(source), bundle=bundle, base_args=base_args,
            ):
                rows.append(_source_row_from(typed, owner=owner))
                if isinstance(typed, CommitRow):
                    commit_shas.append(typed.sha)
                stamp = getattr(typed, "updated_at", None) or getattr(
                    typed, "committed_date", None)
                if stamp and (max_since is None or stamp > max_since):
                    max_since = stamp

        # Check-runs (the fourth entity) have no repo-wide list: they hang off a
        # commit ref, so fan out per fetched commit sha. Skipped when the source
        # has no binding for the kind. Each check-run row keys/refs on its own id.
        cr_bundle = self._transport_provider(source, ENTITY_CHECK_RUN)
        if cr_bundle is not None:
            for sha in commit_shas:
                for typed in self._walk_entity_rows(
                    entity_type=ENTITY_CHECK_RUN, repo=repo,
                    source_id=_source_id_of(source), bundle=cr_bundle,
                    base_args={"owner": owner, "repo": name, "ref": sha},
                ):
                    rows.append(_source_row_from(typed, owner=owner))
                    stamp = getattr(typed, "completed_at", None)
                    if stamp and (max_since is None or stamp > max_since):
                        max_since = stamp
        # Incremental (snapshot=False): a since-window carries only changed rows,
        # so absent rows must NOT be deleted. Checkpoint = the advanced watermark.
        return rows, False, max_since

    def validate_config(self, config: dict) -> tuple[bool, str]:
        # The sources-schema key every connector reads is ``uri`` (as
        # ``local_folder`` does); a ``github://owner/repo`` form is normalised to
        # the bare repo. One spelling only, to match the existing connectors.
        repo = (config.get("uri") or "").strip().removeprefix("github://")
        if not repo:
            return False, "GitHub repository (owner/name) is required"
        if not _REPO_FULL_NAME_RE.match(repo):
            return False, f"repo must be 'owner/name', got {repo!r}"
        return True, ""

    def _walk_entity_rows(
        self, *, entity_type: str, repo: str, source_id: str, bundle: "GithubTransport",
        base_args: "Mapping[str, Any]",
    ) -> tuple:
        """Drive a REAL W01 page walk for one entity and convert its rows.

        Opens a W01 ``PageWalk`` for the entity's operation through the injected
        transport and pumps it with ``walk_pages`` -- so authorization runs per
        page and the walk advances on the SINGLE ``next_cursor`` W01 puts on the
        envelope, never a local re-implementation and never a vendor ``Link``
        parsed here. Each page's rows are read off ``ExecutionOutcome.payload``
        (a :class:`CollectionPayload`) -- the ONE neutral data channel -- and
        handed to PR-2's converter for this kind. A page a gate denied or the
        transport failed on stops the walk and surfaces the typed error rather
        than a partial dataset. Reads the payload; never copies or caches it.
        """

        from kiro_crew.connections.control_plane.result import CollectionPayload
        from kiro_crew.connections.vendors.github.dispatch import (
            open_page_walk,
            walk_pages,
        )

        converter = _CONVERTER_FOR_ENTITY[entity_type]
        fetched_at = _now_iso()
        walk = open_page_walk(
            operation_id=_OP_ALL[entity_type],
            handle=bundle.handle,
            transport=bundle.transport,
            offered_mode=bundle.offered_mode,
            permitted=bundle.permitted,
            layers=bundle.layers,
            governance_scope=bundle.governance_scope,
            governance_item=bundle.governance_item,
            base_args=dict(base_args),
            clock=bundle.clock,
        )
        outcomes = walk_pages(walk)
        rows: list = []
        for outcome in outcomes:
            if not outcome.ok:
                raise LiveFetchError(
                    f"github {entity_type} walk stopped: {outcome.error}")
            payload = outcome.payload
            if payload is None:
                continue  # an empty page: no rows, not an error
            if not isinstance(payload, CollectionPayload):
                # A list operation must return a collection; anything else is a
                # shaping fault, not a silently-empty page.
                raise LiveFetchError(
                    f"github {entity_type} page returned a non-collection payload "
                    f"({type(payload).__name__}); refusing to guess rows")
            for item in payload.items:
                rows.append(
                    converter(
                        repo, dict(item), source_id=source_id, fetched_at=fetched_at))
        return tuple(rows)

    async def detect_changes(self, source: dict) -> bool:
        """Real change detection through W01's transport.

        Fail-closed without a transport (:class:`NotImplementedError`) -- a mock
        cannot answer "changed". With a provider it drives a REAL page walk of the
        pull-request stream filtered by the checkpoint's ``since`` watermark and
        reports whether the window returned any rows: a non-empty ``since`` window
        means at least one record changed after the watermark. The rows come off
        ``ExecutionOutcome.payload``; the walk advances on W01's single
        ``next_cursor``. It never fabricates a "changed" answer.
        """

        if self._transport_provider is None:
            raise NotImplementedError(
                "GitHub live change-detection needs a transport provider that "
                "composes W01's executor/production; none was configured")
        repo = _repo_of(source)
        bundle = self._transport_provider(source, ENTITY_PULL_REQUEST)
        if bundle is None:
            raise LiveFetchError(
                "no W01 transport available for this source; cannot detect changes")
        source_id = _source_id_of(source)
        checkpoint = read_checkpoint(source)
        owner, name = repo.split("/", 1)
        base_args: dict = {"owner": owner, "repo": name}
        if checkpoint.since:
            base_args["since"] = checkpoint.since
        rows = self._walk_entity_rows(
            entity_type=ENTITY_PULL_REQUEST, repo=repo, source_id=source_id,
            bundle=bundle, base_args=base_args)
        return len(rows) > 0

    async def fetch(self, source: dict) -> tuple[str, dict]:
        """A REAL structured fetch + refresh through W01's transport.

        Fail-closed without a transport (:class:`NotImplementedError`). With a
        provider it walks each wired entity kind in :data:`_OP_FOR_ENTITY` through
        W01's transport (the operation is invoked, authorized per page, and the
        walk advances on W01's single ``next_cursor``), reads the rows off
        ``ExecutionOutcome.payload``, converts them with PR-2's converters, folds
        the result through PR-2's ``diff_rows`` (a ``since`` window, so
        ``disappeared`` stays empty), and renders the upserts into the pipeline's
        ``(text, metadata)`` shape. Every stored item carries the full lineage
        domain, so a search hit stays traceable.

        Returns one text blob joining every row's projection and a metadata dict
        carrying the row count, the next watermark to persist, and the per-row
        primary keys -- the scheduler chunks and stores it as for any other
        source.
        """

        if self._transport_provider is None:
            raise NotImplementedError(
                "GitHub live fetch needs a transport provider that composes "
                "W01's executor/production and consumes vendors.github paging; "
                "none was configured")
        repo = _repo_of(source)
        source_id = _source_id_of(source)
        checkpoint = read_checkpoint(source)
        owner, name = repo.split("/", 1)
        fetched_rows: list = []
        for entity_type in _OP_FOR_ENTITY:
            bundle = self._transport_provider(source, entity_type)
            if bundle is None:
                continue  # this source does not read this kind
            base_args: dict = {"owner": owner, "repo": name}
            if checkpoint.since:
                base_args["since"] = checkpoint.since
            fetched_rows.extend(
                self._walk_entity_rows(
                    entity_type=entity_type, repo=repo, source_id=source_id,
                    bundle=bundle, base_args=base_args))

        # A windowed (since) refresh never reports disappearances; a first full
        # refresh also passes full_listing=False, because a multi-entity walk is
        # not a single-collection full listing -- retiring is out of scope here.
        plan = diff_rows(
            tuple(fetched_rows), frozenset(),
            prior_since=checkpoint.since, full_listing=False)
        text = "\n\n".join(render_row_text(row) for row in plan.upserts)
        metadata = {
            "source_type": SOURCE_TYPE,
            "repo_full_name": repo,
            "row_count": len(plan.upserts),
            "next_since": plan.next_since,
            "primary_keys": [row.primary_key for row in plan.upserts],
            "rows": [render_row_metadata(row) for row in plan.upserts],
        }
        return text, metadata
