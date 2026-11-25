# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

"""Unit & integration tests for async crawler using httpx.MockTransport (no live network)."""

import asyncio
import random
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler import (
    DEFAULT_USER_AGENT,
    CrawledPage,
    CrawlStats,
    RobotsCache,
    canonicalize_url,
    crawl,
    crawl_to_jsonl,
    extract_text_and_links,
    fetch,
    parse_jsonl_line,
    parse_retry_after,
)
from findex.index import build_index
from findex.search import search_ranked


def test_canonicalize_url_normalization() -> None:
    assert (
        canonicalize_url("HTTP://Docs.Example.COM:80/guide//intro/?b=2&a=1#section-1")
        == "http://docs.example.com/guide/intro?a=1&b=2"
    )
    assert canonicalize_url("https://example.com:443/docs/") == "https://example.com/docs"
    assert (
        canonicalize_url("../api/ref/?z=9&a=1#top", base_url="https://example.com/docs/v1/index")
        == "https://example.com/docs/api/ref?a=1&z=9"
    )
    assert canonicalize_url("mailto:admin@example.com") is None
    assert canonicalize_url("javascript:void(0)") is None
    assert canonicalize_url("#only-fragment") is None
    assert canonicalize_url("") is None
    assert canonicalize_url("http://") is None


def test_parse_retry_after_and_jsonl_helper(tmp_path: Path) -> None:
    assert parse_retry_after(None) == 0.0
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
    assert parse_retry_after("invalid-date") == 0.0

    sample_path = tmp_path / "sample.jsonl"
    doc_id, text, path = parse_jsonl_line(
        '{"url": "https://ex.com/1", "text": "hello"}',
        fallback_id="fb",
        source_path=sample_path,
    )
    assert doc_id == "https://ex.com/1"
    assert text == "hello"
    assert path == sample_path


def test_extract_text_and_links_ignores_scripts_and_deduplicates() -> None:
    html = """
    <html>
      <head>
        <title>Asyncio Guide</title>
        <script>const secret = "should_not_appear";</script>
        <style>body { color: red; }</style>
      </head>
      <body>
        <h1>Event Loop</h1>
        <p>Coroutines use await.</p>
        <a href="/docs/tasks/#part1">Tasks</a>
        <a href="https://example.com/docs/tasks">Duplicate</a>
        <a href="mailto:test@example.com">Email</a>
      </body>
    </html>
    """
    title, text, links = extract_text_and_links(html, base_url="https://example.com/docs/start")
    assert title == "Asyncio Guide"
    assert "Event Loop" in text
    assert "Coroutines use await." in text
    assert "should_not_appear" not in text
    assert links == ["https://example.com/docs/tasks"]


def test_fetch_retries_429_and_5xx_with_retry_after_and_skips_404() -> None:
    async def _run() -> None:
        calls: dict[str, int] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls[path] = calls.get(path, 0) + 1
            if path == "/rate-limited":
                if calls[path] < 3:
                    return httpx.Response(429, headers={"Retry-After": "0.02"}, text="Slow down")
                return httpx.Response(200, text="Recovered after 429")
            if path == "/flaky-503":
                if calls[path] < 4:
                    return httpx.Response(503, text="Service unavailable")
                return httpx.Response(200, text="Recovered after 503")
            if path == "/missing":
                return httpx.Response(404, text="Not found")
            if path == "/loop-1":
                return httpx.Response(302, headers={"Location": "/loop-2"})
            if path == "/loop-2":
                return httpx.Response(302, headers={"Location": "/loop-1"})
            if path == "/conn-err":
                raise httpx.ConnectError("Connection refused", request=request)
            return httpx.Response(200, text="OK")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            res_429 = await fetch(
                client,
                "https://example.com/rate-limited",
                base_backoff=0.01,
                rng=random.Random(1),
            )
            assert res_429.status_code == 200
            assert res_429.attempts == 3
            assert res_429.content == b"Recovered after 429"
            assert res_429.bytes_received == len(b"Recovered after 429")
            assert res_429.elapsed_seconds > 0.0

            res_503 = await fetch(
                client,
                "https://example.com/flaky-503",
                max_retries=3,
                base_backoff=0.005,
                rng=random.Random(1),
            )
            assert res_503.status_code == 200
            assert res_503.attempts == 4

            res_404 = await fetch(
                client,
                "https://example.com/missing",
                max_retries=3,
                base_backoff=0.01,
            )
            assert res_404.status_code == 404
            assert res_404.attempts == 1
            assert calls["/missing"] == 1

            res_loop = await fetch(client, "https://example.com/loop-1", max_retries=3)
            assert res_loop.status_code == 0
            assert res_loop.attempts == 1
            assert res_loop.error is not None and "TooManyRedirects" in res_loop.error

            res_conn = await fetch(client, "https://example.com/conn-err", max_retries=3)
            assert res_conn.status_code == 0
            assert res_conn.attempts == 1
            assert res_conn.error is not None and "ConnectError" in res_conn.error

    asyncio.run(_run())


def test_fetch_retries_timeout_up_to_three_times() -> None:
    async def _run() -> None:
        attempts_seen = 0

        async def slow_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts_seen
            attempts_seen += 1
            await asyncio.sleep(0.2)
            return httpx.Response(200, text="Too late")

        transport = httpx.MockTransport(slow_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/hang",
                timeout=0.02,
                max_retries=3,
                base_backoff=0.005,
            )
            assert res.status_code == 0
            assert res.error == "TimeoutError"
            assert res.attempts == 4
            assert attempts_seen == 4

    asyncio.run(_run())


def test_robots_cache_fetches_once_per_host_and_enforces_rules() -> None:
    async def _run() -> None:
        robots_hits: dict[str, int] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if request.url.path == "/robots.txt":
                robots_hits[host] = robots_hits.get(host, 0) + 1
                if host == "alpha.test":
                    return httpx.Response(
                        200,
                        text="User-agent: *\nDisallow: /private/\nCrawl-delay: 1\n",
                    )
                if host == "forbidden.test":
                    return httpx.Response(403, text="Forbidden")
                if host == "broken.test":
                    raise httpx.ConnectError("Robots unreachable", request=request)
                return httpx.Response(404, text="No robots")
            return httpx.Response(200, text="Page")

        transport = httpx.MockTransport(handler)
        cache = RobotsCache()
        async with httpx.AsyncClient(transport=transport) as client:
            assert await cache.can_fetch(client, "https://alpha.test/public/a", DEFAULT_USER_AGENT)
            assert cache.is_cached("https://alpha.test/other")
            assert cache.get_crawl_delay("https://alpha.test/public/a", DEFAULT_USER_AGENT) == 1.0
            assert not await cache.can_fetch(
                client, "https://alpha.test/private/secret", DEFAULT_USER_AGENT
            )
            assert await cache.can_fetch(client, "https://beta.test/anything", DEFAULT_USER_AGENT)
            assert not await cache.can_fetch(
                client, "https://forbidden.test/page", DEFAULT_USER_AGENT
            )
            assert await cache.can_fetch(client, "https://broken.test/page", DEFAULT_USER_AGENT)

        assert robots_hits["alpha.test"] == 1
        assert robots_hits["beta.test"] == 1
        assert robots_hits["forbidden.test"] == 1
        assert robots_hits["broken.test"] == 1

    asyncio.run(_run())


def test_crawl_generator_respects_max_pages_robots_and_streams_to_index(tmp_path: Path) -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
            if path == "/start":
                body = (
                    "<title>Start</title><p>Python asyncio crawler</p>"
                    '<a href="/p1">P1</a><a href="/p2#frag">P2</a>'
                    '<a href="/blocked">Blocked</a><a href="/file.bin">Bin</a>'
                    '<a href="/start?">Dup</a>'
                )
                return httpx.Response(200, headers={"content-type": "text/html"}, text=body)
            if path == "/file.bin":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/octet-stream"},
                    content=b"\x00\x01\x02",
                )
            if path in {"/p1", "/p2", "/p3", "/p4"}:
                idx_num = int(path[2:])
                body = (
                    f"<title>Page {idx_num}</title>"
                    f"<p>Document number {idx_num} about coroutines and queues</p>"
                    f'<a href="/p{idx_num + 1}">Next</a>'
                )
                return httpx.Response(200, headers={"content-type": "text/html"}, text=body)
            return httpx.Response(404, text="Not found")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        stats = CrawlStats()
        pages: list[CrawledPage] = []
        async for page in crawl(
            "https://docs.test/start",
            max_pages=3,
            concurrency=4,
            per_host_concurrency=2,
            min_delay=0.01,
            client=client,
            stats=stats,
        ):
            pages.append(page)

        await client.aclose()
        assert len(pages) == 3
        urls = [p.url for p in pages]
        assert "https://docs.test/blocked" not in urls
        assert len(set(urls)) == 3

        jsonl_path = tmp_path / "pages.jsonl"
        log_path = tmp_path / "crawl.log"
        client2 = httpx.AsyncClient(transport=transport)
        await crawl_to_jsonl(
            "https://docs.test/start",
            jsonl_path,
            log_path=log_path,
            max_pages=4,
            concurrency=3,
            min_delay=0.005,
            client=client2,
        )
        await client2.aclose()

        streamed_docs = list(iter_documents(jsonl_path))
        assert len(streamed_docs) == 4
        assert log_path.exists()
        assert "STATUS=200" in log_path.read_text(encoding="utf-8")

        built_idx = build_index(jsonl_path, workers=1, executor="serial")
        assert built_idx.num_docs == 4
        hits = search_ranked(built_idx, "coroutines")
        assert len(hits) >= 1

    asyncio.run(_run())


def test_cli_crawl_and_index_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<title>CLI Test</title><p>Asyncio event loop indexing</p>",
        )

    transport = httpx.MockTransport(handler)
    orig_init = httpx.AsyncClient.__init__

    def _patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_init)

    runner = CliRunner()
    jsonl_out = tmp_path / "out.jsonl"
    log_out = tmp_path / "crawl.log"
    idx_out = tmp_path / "web_index.json"

    res_crawl = runner.invoke(
        app,
        [
            "crawl",
            "https://local.test/home",
            "-o",
            str(jsonl_out),
            "--log",
            str(log_out),
            "--max-pages",
            "1",
            "--concurrency",
            "2",
            "--delay",
            "0",
            "--debug",
        ],
    )
    assert res_crawl.exit_code == 0
    assert jsonl_out.exists()
    assert log_out.exists()

    res_idx = runner.invoke(
        app,
        ["index", str(jsonl_out), "-o", str(idx_out), "--workers", "1", "--executor", "serial"],
    )
    assert res_idx.exit_code == 0

    res_search = runner.invoke(app, ["search", str(idx_out), "event loop"])
    assert res_search.exit_code == 0
    assert "local.test/home" in res_search.stdout
