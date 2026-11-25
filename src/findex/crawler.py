"""Asynchronous polite web crawler built on asyncio, TaskGroup, and httpx."""

import asyncio
import json
import logging
import posixpath
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

DEFAULT_USER_AGENT = (
    "findex-crawler/0.6.0 "
    "(+https://github.com/findex-project/findex; contact: crawler@findex.local)"
)

_IGNORED_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})


def _empty_headers() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single HTTP fetch (including retry attempts)."""

    url: str
    final_url: str
    status_code: int
    content: bytes
    elapsed_seconds: float
    attempts: int
    headers: dict[str, str] = field(default_factory=_empty_headers)
    error: str | None = None

    @property
    def bytes_received(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class CrawledPage:
    """Successfully fetched and parsed HTML page yielded by crawl()."""

    url: str
    title: str
    text: str
    links: list[str]
    status_code: int
    elapsed_seconds: float
    bytes_received: int


@dataclass
class CrawlStats:
    """Live telemetry counters exposed to Rich UI and benchmarks."""

    pages_collected: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return max(time.monotonic() - self.start_time, 1e-6)

    @property
    def pages_per_second(self) -> float:
        return self.pages_collected / self.elapsed_seconds


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """
    Normalize a URL for deduplication:
    - Resolve relative URLs against base_url when provided
    - Keep only http/https schemes
    - Lowercase scheme and host; strip default ports (:80, :443)
    - Drop #fragment
    - Normalize path slashes and dot segments
    - Sort query parameters deterministically
    """
    raw = url.strip()
    if not raw or raw.startswith("#"):
        return None

    joined = urljoin(base_url, raw) if base_url is not None else raw
    parts = urlsplit(joined)

    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    hostname = (parts.hostname or "").lower()
    if not hostname:
        return None

    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname

    raw_path = parts.path or "/"
    normalized_path = posixpath.normpath(raw_path)
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path.rstrip("/")

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    query_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    normalized_query = urlencode(query_pairs, doseq=True)

    return urlunsplit((scheme, netloc, normalized_path, normalized_query, ""))


class _HtmlLinkAndTextParser(HTMLParser):
    """Stdlib html.parser extractor for visible text, page title, and anchor hrefs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag in _IGNORED_TAGS:
            self._skip_depth += 1
            return
        if lower_tag == "title":
            self._in_title = True
        if lower_tag == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val is not None:
                    self.hrefs.append(attr_val)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in _IGNORED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lower_tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


def extract_text_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Extract (title, visible_text, canonical_links) from HTML markup."""
    parser = _HtmlLinkAndTextParser()
    parser.feed(html)
    parser.close()

    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()

    links: list[str] = []
    seen_local: set[str] = set()
    for href in parser.hrefs:
        canon = canonicalize_url(href, base_url=base_url)
        if canon is not None and canon not in seen_local:
            seen_local.add(canon)
            links.append(canon)

    return title, text, links


def parse_retry_after(header_value: str | None) -> float:
    """Parse HTTP Retry-After header (seconds or HTTP-date) into non-negative delay seconds."""
    if not header_value:
        return 0.0
    stripped = header_value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        target_dt = parsedate_to_datetime(stripped)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=UTC)
        delta = (target_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return 0.0


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.5,
    max_backoff: float = 15.0,
    jitter_ratio: float = 0.25,
    rng: random.Random | None = None,
    url_logger: logging.Logger | None = None,
) -> FetchResult:
    """
    Fetch a URL with asyncio.timeout and bounded retries (up to max_retries).
    Retries ONLY on HTTP 429, HTTP 5xx, and timeouts with exponential backoff + jitter
    while respecting Retry-After. Never retries other 4xx codes or redirect loops.
    """
    rand = rng if rng is not None else random.Random()
    t_overall_start = time.perf_counter()
    last_status = 0
    last_headers: dict[str, str] = {}
    last_error: str | None = None
    final_url = url

    for attempt in range(1, max_retries + 2):
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url)

            last_status = response.status_code
            last_headers = {k.lower(): v for k, v in response.headers.items()}
            final_url = str(response.url)

            if last_status == 429 or 500 <= last_status < 600:
                last_error = f"HTTP {last_status}"
                if attempt <= max_retries:
                    exp_delay = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                    jitter = rand.uniform(0.0, exp_delay * jitter_ratio)
                    retry_after = parse_retry_after(last_headers.get("retry-after"))
                    delay = max(retry_after, exp_delay + jitter)
                    if url_logger is not None:
                        url_logger.warning(
                            "STATUS=%d | ATTEMPT=%d/%d | RETRY_IN=%.2fs | REASON=%s | URL=%s",
                            last_status,
                            attempt,
                            max_retries + 1,
                            delay,
                            last_error,
                            url,
                        )
                    await asyncio.sleep(delay)
                    continue

            elapsed = time.perf_counter() - t_overall_start
            err_msg = f"HTTP {last_status}" if last_status >= 400 else None
            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=last_status,
                content=response.content,
                elapsed_seconds=elapsed,
                attempts=attempt,
                headers=last_headers,
                error=err_msg,
            )

        except (TimeoutError, httpx.TimeoutException):
            last_status = 0
            last_error = "TimeoutError"
            if attempt <= max_retries:
                exp_delay = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                jitter = rand.uniform(0.0, exp_delay * jitter_ratio)
                delay = exp_delay + jitter
                if url_logger is not None:
                    url_logger.warning(
                        "STATUS=TIMEOUT | ATTEMPT=%d/%d | RETRY_IN=%.2fs | URL=%s",
                        attempt,
                        max_retries + 1,
                        delay,
                        url,
                    )
                await asyncio.sleep(delay)
                continue

            elapsed = time.perf_counter() - t_overall_start
            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=0,
                content=b"",
                elapsed_seconds=elapsed,
                attempts=attempt,
                headers=last_headers,
                error=last_error,
            )

        except httpx.TooManyRedirects as exc:
            elapsed = time.perf_counter() - t_overall_start
            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=0,
                content=b"",
                elapsed_seconds=elapsed,
                attempts=attempt,
                headers=last_headers,
                error=f"TooManyRedirects: {exc}",
            )

        except httpx.RequestError as exc:
            elapsed = time.perf_counter() - t_overall_start
            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=0,
                content=b"",
                elapsed_seconds=elapsed,
                attempts=attempt,
                headers=last_headers,
                error=f"{type(exc).__name__}: {exc}",
            )

    elapsed = time.perf_counter() - t_overall_start
    return FetchResult(
        url=url,
        final_url=final_url,
        status_code=last_status,
        content=b"",
        elapsed_seconds=elapsed,
        attempts=max_retries + 1,
        headers=last_headers,
        error=last_error or "Max retries exceeded",
    )


@dataclass
class _HostRobotsEntry:
    parser: RobotFileParser
    policy: Literal["allow_all", "disallow_all", "parsed"] = "parsed"


class RobotsCache:
    """Fetches, parses, and caches /robots.txt rules separately per host."""

    def __init__(self, timeout: float = 5.0) -> None:
        self._timeout = timeout
        self._cache: dict[str, _HostRobotsEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _host_origin(url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"

    def is_cached(self, url: str) -> bool:
        """Return True if robots.txt for the URL's host is already cached."""
        return self._host_origin(url) in self._cache

    async def _load_for_origin(self, client: httpx.AsyncClient, origin: str) -> _HostRobotsEntry:
        robots_url = f"{origin}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            async with asyncio.timeout(self._timeout):
                resp = await client.get(robots_url)
            if resp.status_code in {401, 403}:
                return _HostRobotsEntry(parser=rp, policy="disallow_all")
            if resp.status_code >= 400:
                return _HostRobotsEntry(parser=rp, policy="allow_all")
            rp.parse(resp.text.splitlines())
            rp.modified()
            return _HostRobotsEntry(parser=rp, policy="parsed")
        except Exception:
            return _HostRobotsEntry(parser=rp, policy="allow_all")

    async def can_fetch(
        self,
        client: httpx.AsyncClient,
        url: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> bool:
        """Check whether user_agent is allowed to fetch url according to cached robots.txt."""
        origin = self._host_origin(url)
        entry = self._cache.get(origin)
        if entry is None:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                entry = self._cache.get(origin)
                if entry is None:
                    entry = await self._load_for_origin(client, origin)
                    self._cache[origin] = entry

        if entry.policy == "disallow_all":
            return False
        if entry.policy == "allow_all":
            return True
        return entry.parser.can_fetch(user_agent, url)

    def get_crawl_delay(self, url: str, user_agent: str = DEFAULT_USER_AGENT) -> float | None:
        """Return Crawl-delay directive in seconds if specified by cached robots.txt."""
        origin = self._host_origin(url)
        entry = self._cache.get(origin)
        if entry is None or entry.policy != "parsed":
            return None
        raw_delay = entry.parser.crawl_delay(user_agent)
        if isinstance(raw_delay, (int, float)):
            return float(raw_delay)
        if isinstance(raw_delay, str):
            try:
                return float(raw_delay)
            except ValueError:
                return None
        return None


async def crawl(
    start_urls: str | list[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host_concurrency: int = 2,
    min_delay: float = 0.2,
    timeout: float = 10.0,
    max_retries: int = 3,
    user_agent: str = DEFAULT_USER_AGENT,
    same_host_only: bool = True,
    client: httpx.AsyncClient | None = None,
    robots_cache: RobotsCache | None = None,
    stats: CrawlStats | None = None,
    url_logger: logging.Logger | None = None,
) -> AsyncIterator[CrawledPage]:
    """
    Asynchronous generator crawling web pages concurrently with:
    - URL frontier queue & seen canonical URL deduplication
    - asyncio.TaskGroup worker pool
    - Global concurrency Semaphore + per-host Semaphore + per-host minimum delay
    - Per-host RobotsCache enforcement
    - Immediate worker cancellation and client cleanup upon reaching max_pages
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if per_host_concurrency < 1:
        raise ValueError("per_host_concurrency must be >= 1")

    seeds = [start_urls] if isinstance(start_urls, str) else list(start_urls)
    canonical_seeds: list[str] = []
    allowed_hosts: set[str] = set()
    for seed in seeds:
        canon = canonicalize_url(seed)
        if canon is not None:
            canonical_seeds.append(canon)
            allowed_hosts.add(urlsplit(canon).netloc.lower())

    if not canonical_seeds:
        return

    owns_client = client is None
    active_client = (
        client
        if client is not None
        else httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            max_redirects=10,
        )
    )
    robots = robots_cache if robots_cache is not None else RobotsCache(timeout=min(timeout, 5.0))
    live_stats = stats if stats is not None else CrawlStats()
    live_stats.start_time = time.monotonic()

    frontier: asyncio.Queue[str] = asyncio.Queue()
    results_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    seen: set[str] = set()

    for canon_seed in canonical_seeds:
        if canon_seed not in seen:
            seen.add(canon_seed)
            frontier.put_nowait(canon_seed)

    live_stats.queue_depth = frontier.qsize()
    pending_urls = frontier.qsize()
    stop_event = asyncio.Event()

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    host_last_request: dict[str, float] = {}

    def _enqueue_url(candidate: str) -> None:
        nonlocal pending_urls
        if stop_event.is_set() or len(seen) >= max_pages * 5:
            return
        if same_host_only and urlsplit(candidate).netloc.lower() not in allowed_hosts:
            return
        if candidate not in seen:
            seen.add(candidate)
            pending_urls += 1
            frontier.put_nowait(candidate)
            live_stats.queue_depth = frontier.qsize()

    def _complete_one_url() -> None:
        nonlocal pending_urls
        pending_urls -= 1
        live_stats.queue_depth = frontier.qsize()
        if pending_urls <= 0 and not stop_event.is_set():
            stop_event.set()
            results_queue.put_nowait(None)

    async def _worker() -> None:
        while not stop_event.is_set():
            url = await frontier.get()
            live_stats.queue_depth = frontier.qsize()
            try:
                if stop_event.is_set():
                    return

                allowed = await robots.can_fetch(active_client, url, user_agent=user_agent)
                if not allowed:
                    if url_logger is not None:
                        url_logger.info("STATUS=ROBOTS_DISALLOWED | URL=%s", url)
                    continue

                host = urlsplit(url).netloc.lower()
                host_sem = host_sems.setdefault(host, asyncio.Semaphore(per_host_concurrency))
                rate_lock = host_rate_locks.setdefault(host, asyncio.Lock())
                robots_delay = robots.get_crawl_delay(url, user_agent=user_agent)
                effective_delay = max(min_delay, robots_delay or 0.0)

                async with global_sem, host_sem:
                    if stop_event.is_set():
                        return

                    if effective_delay > 0.0:
                        async with rate_lock:
                            now = time.monotonic()
                            last_ts = host_last_request.get(host, 0.0)
                            wait_time = effective_delay - (now - last_ts)
                            if wait_time > 0.0:
                                await asyncio.sleep(wait_time)
                            host_last_request[host] = time.monotonic()

                    if stop_event.is_set():
                        return

                    live_stats.in_flight += 1
                    try:
                        res = await fetch(
                            active_client,
                            url,
                            timeout=timeout,
                            max_retries=max_retries,
                            url_logger=url_logger,
                        )
                    finally:
                        live_stats.in_flight -= 1

                if res.error is not None or res.status_code >= 400:
                    live_stats.errors += 1
                    if url_logger is not None:
                        url_logger.error(
                            "STATUS=%s | ATTEMPTS=%d | TIME=%.3fs | ERROR=%s | URL=%s",
                            res.status_code if res.status_code else "ERR",
                            res.attempts,
                            res.elapsed_seconds,
                            res.error,
                            url,
                        )
                    continue

                if url_logger is not None:
                    url_logger.info(
                        "STATUS=%d | ATTEMPTS=%d | TIME=%.3fs | BYTES=%d | URL=%s",
                        res.status_code,
                        res.attempts,
                        res.elapsed_seconds,
                        res.bytes_received,
                        url,
                    )

                content_type = res.headers.get("content-type", "text/html").lower()
                if "html" not in content_type and "xml" not in content_type:
                    continue

                html_text = res.content.decode("utf-8", errors="replace")
                title, visible_text, discovered_links = extract_text_and_links(
                    html_text,
                    base_url=res.final_url or url,
                )

                for link in discovered_links:
                    _enqueue_url(link)

                page = CrawledPage(
                    url=url,
                    title=title,
                    text=visible_text,
                    links=discovered_links,
                    status_code=res.status_code,
                    elapsed_seconds=res.elapsed_seconds,
                    bytes_received=res.bytes_received,
                )
                if not stop_event.is_set():
                    results_queue.put_nowait(page)
            finally:
                frontier.task_done()
                _complete_one_url()

    async def _run_task_group() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(concurrency):
                    tg.create_task(_worker())
        except asyncio.CancelledError:
            pass

    coordinator = asyncio.create_task(_run_task_group())
    yielded_count = 0

    try:
        while yielded_count < max_pages:
            item = await results_queue.get()
            if item is None:
                break
            yielded_count += 1
            live_stats.pages_collected = yielded_count
            yield item
    finally:
        stop_event.set()
        if not coordinator.done():
            coordinator.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)
        if owns_client:
            await active_client.aclose()


def configure_crawl_logger(log_path: Path) -> logging.Logger:
    """Create a dedicated file logger writing per-URL status entries to crawl.log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    crawl_log = logging.getLogger(f"findex.crawl.{log_path.resolve()}")
    crawl_log.handlers.clear()
    crawl_log.propagate = False
    crawl_log.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    crawl_log.addHandler(file_handler)
    return crawl_log


async def crawl_to_jsonl(
    start_urls: str | list[str],
    output_jsonl: Path,
    *,
    log_path: Path = Path("crawl.log"),
    max_pages: int = 200,
    concurrency: int = 10,
    per_host_concurrency: int = 2,
    min_delay: float = 0.2,
    timeout: float = 10.0,
    max_retries: int = 3,
    user_agent: str = DEFAULT_USER_AGENT,
    same_host_only: bool = True,
    client: httpx.AsyncClient | None = None,
    robots_cache: RobotsCache | None = None,
    stats: CrawlStats | None = None,
    on_page: Callable[[CrawledPage, CrawlStats], None] | None = None,
) -> CrawlStats:
    """
    Stream crawled pages directly to a JSONL file as they arrive without holding all pages in RAM.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    live_stats = stats if stats is not None else CrawlStats()
    url_logger = configure_crawl_logger(log_path)

    try:
        with output_jsonl.open("w", encoding="utf-8") as jsonl_file:
            async for page in crawl(
                start_urls,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_concurrency=per_host_concurrency,
                min_delay=min_delay,
                timeout=timeout,
                max_retries=max_retries,
                user_agent=user_agent,
                same_host_only=same_host_only,
                client=client,
                robots_cache=robots_cache,
                stats=live_stats,
                url_logger=url_logger,
            ):
                record = {
                    "doc_id": page.url,
                    "url": page.url,
                    "title": page.title,
                    "text": f"{page.title}\n{page.text}".strip() if page.title else page.text,
                    "status_code": page.status_code,
                    "elapsed_seconds": round(page.elapsed_seconds, 4),
                    "bytes_received": page.bytes_received,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()
                if on_page is not None:
                    on_page(page, live_stats)
    finally:
        for handler in list(url_logger.handlers):
            handler.close()
            url_logger.removeHandler(handler)

    return live_stats


def parse_jsonl_line(raw_line: str, fallback_id: str, source_path: Path) -> tuple[str, str, Path]:
    """Helper to decode one JSONL line into (doc_id, text, path)."""
    obj = cast(dict[str, object], json.loads(raw_line))
    doc_id = str(obj.get("doc_id") or obj.get("url") or fallback_id)
    text = str(obj.get("text", ""))
    return doc_id, text, source_path
