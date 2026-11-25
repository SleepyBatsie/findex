"""Typer + Rich command-line interface for findex."""

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.corpus import discover_files
from findex.crawler import DEFAULT_USER_AGENT, CrawlStats, crawl_to_jsonl
from findex.index import ExecutorName, build_index
from findex.ranking import ScorerName
from findex.search import search_ast, search_ranked
from findex.semantic import (
    DEFAULT_EMBED_MODEL,
    SearchMode,
    SemanticIndex,
    build_semantic_index,
    search_hybrid,
    search_semantic,
)
from findex.stats import compute_stats
from findex.storage import load_index, save_index
from findex.tokenize import tokenize
from findex.utils import generate_snippet

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="Full-text and semantic search engine CLI with web UI, async crawler, and NumPy scoring.",
    no_args_is_help=True,
)


def configure_logging(verbose: int) -> None:
    """Configure package logging once at CLI startup, routing diagnostics to stderr."""
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    pkg_logger = logging.getLogger("findex")
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = True


def _fail(message: str) -> NoReturn:
    """Print a single clean error line to stderr and exit with code 1."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _render_crawl_dashboard(stats: CrawlStats, max_pages: int, start_url: str) -> Table:
    """Build a live Rich status table showing crawler progress metrics."""
    table = Table(title=f"findex crawl: {start_url}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Pages Collected", f"{stats.pages_collected} / {max_pages}")
    table.add_row("Pages / sec", f"{stats.pages_per_second:.2f}")
    table.add_row("In-Flight Requests", str(stats.in_flight))
    table.add_row("Errors", str(stats.errors))
    table.add_row("Queue Depth", str(stats.queue_depth))
    return table


def _lookup_doc_text(doc_file: Path, doc_id: str) -> str:
    """Retrieve document text for snippet generation from plain text or JSONL files."""
    if not doc_file.is_file():
        return ""
    if doc_file.suffix.lower() != ".jsonl":
        return doc_file.read_text(encoding="utf-8", errors="replace")
    with doc_file.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = cast(dict[str, object], json.loads(line))
            if str(obj.get("doc_id") or obj.get("url") or "") == doc_id:
                return str(obj.get("text", ""))
    return ""


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase diagnostic logging verbosity (-v for INFO, -vv for DEBUG).",
        ),
    ] = 0,
) -> None:
    """Global CLI options."""
    configure_logging(verbose)


@app.command("serve")
def cmd_serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host address."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Bind port number."),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Number of Uvicorn worker processes."),
    ] = 1,
    index_path: Annotated[
        Path | None,
        typer.Option("--index", "-i", help="Optional path to index.json (overrides INDEX_PATH)."),
    ] = None,
    embeddings_path: Annotated[
        Path | None,
        typer.Option("--embeddings", "-e", help="Optional path to embeddings.npy."),
    ] = None,
) -> None:
    """Start the FastAPI web search server via Uvicorn."""
    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)
    if embeddings_path is not None:
        os.environ["EMBEDDINGS_PATH"] = str(embeddings_path)

    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
    )


@app.command("crawl")
def cmd_crawl(
    start_url: Annotated[
        str,
        typer.Argument(help="Seed URL to start polite asynchronous crawling."),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Destination JSONL path for crawled pages."),
    ] = Path("crawled.jsonl"),
    log_file: Annotated[
        Path,
        typer.Option("--log", help="Log file recording per-URL HTTP status and errors."),
    ] = Path("crawl.log"),
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", min=1, help="Maximum number of pages to collect."),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("-c", "--concurrency", min=1, help="Maximum concurrent HTTP requests."),
    ] = 10,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Maximum concurrent requests per host."),
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Minimum delay in seconds between same-host hits."),
    ] = 0.2,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Per-request timeout in seconds."),
    ] = 10.0,
    user_agent: Annotated[
        str,
        typer.Option("--user-agent", help="Honest User-Agent header with contact info."),
    ] = DEFAULT_USER_AGENT,
    same_host: Annotated[
        bool,
        typer.Option("--same-host/--allow-external", help="Restrict crawling to seed host."),
    ] = True,
    debug_loop: Annotated[
        bool,
        typer.Option("--debug", help="Enable asyncio debug mode to detect event loop blocking."),
    ] = False,
) -> None:
    """Politely crawl web pages asynchronously and stream them into JSONL."""
    err_console = Console(stderr=True)
    stats = CrawlStats()

    async def _run() -> None:
        if debug_loop:
            loop = asyncio.get_running_loop()
            loop.set_debug(True)
            loop.slow_callback_duration = 0.1

        done_refresh = asyncio.Event()

        with Live(
            _render_crawl_dashboard(stats, max_pages, start_url),
            console=err_console,
            refresh_per_second=8,
        ) as live:

            async def _refresh_ui() -> None:
                while not done_refresh.is_set():
                    live.update(_render_crawl_dashboard(stats, max_pages, start_url))
                    await asyncio.sleep(0.12)
                live.update(_render_crawl_dashboard(stats, max_pages, start_url))

            ui_task = asyncio.create_task(_refresh_ui())
            try:
                await crawl_to_jsonl(
                    start_url,
                    output,
                    log_path=log_file,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host_concurrency=per_host,
                    min_delay=delay,
                    timeout=timeout,
                    user_agent=user_agent,
                    same_host_only=same_host,
                    stats=stats,
                )
            finally:
                done_refresh.set()
                await ui_task

    try:
        asyncio.run(_run(), debug=debug_loop)
        err_console.print(
            f"Crawled [bold]{stats.pages_collected}[/bold] pages in "
            f"[bold]{stats.elapsed_seconds:.2f}s[/bold] "
            f"([bold]{stats.pages_per_second:.2f}[/bold] pages/s, errors={stats.errors}) "
            f"-> [cyan]{output}[/cyan] (log: [cyan]{log_file}[/cyan])"
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _fail(f"Crawl failed: {exc}")


@app.command("index")
def cmd_index(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Directory or file (including JSONL) containing documents to index."),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Destination JSON path for the built index."),
    ] = Path("index.json"),
    workers: Annotated[
        int,
        typer.Option(
            "-w",
            "--workers",
            min=1,
            help="Number of worker threads or processes.",
        ),
    ] = max(1, os.cpu_count() or 1),
    executor: Annotated[
        ExecutorName,
        typer.Option(
            "--executor",
            help="Execution backend: serial, threads, or processes.",
        ),
    ] = "processes",
) -> None:
    """Build an inverted index from a directory of documents or crawled JSONL."""
    err_console = Console(stderr=True)
    try:
        files = discover_files(corpus_dir)
    except (FileNotFoundError, OSError) as exc:
        _fail(str(exc))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
        ) as progress:
            task = progress.add_task(
                f"Indexing ({executor}, workers={workers})...",
                total=len(files),
            )
            idx = build_index(
                corpus_dir,
                workers=workers,
                executor=executor,
                on_chunk_done=lambda count: progress.advance(task, advance=count),
            )

        saved_path = save_index(idx, output)
        logger.info("Index successfully saved to %s", saved_path)
        err_console.print(
            f"Indexed [bold]{idx.num_docs}[/bold] documents "
            f"([bold]{len(idx)}[/bold] unique terms) -> [cyan]{saved_path}[/cyan]"
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _fail(f"Index build failed: {exc}")


@app.command("embed")
def cmd_embed(
    index_path: Annotated[
        Path,
        typer.Argument(help="Path to the serialized index JSON file."),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Destination .npy path for chunk embeddings."),
    ] = Path("data/embeddings.npy"),
    model_name: Annotated[
        str,
        typer.Option("--model", help="SentenceTransformer model name."),
    ] = DEFAULT_EMBED_MODEL,
    chunk_words: Annotated[
        int,
        typer.Option("--chunk-words", min=20, help="Number of words per document chunk."),
    ] = 120,
    overlap_words: Annotated[
        int,
        typer.Option("--overlap-words", min=0, help="Overlapping words between chunks."),
    ] = 30,
) -> None:
    """Split indexed documents into chunks, compute normalized embeddings, and save .npy."""
    err_console = Console(stderr=True)
    try:
        idx = load_index(index_path)
        doc_texts: list[tuple[str, str]] = []
        for doc_id, meta in sorted(idx.doc_meta.items()):
            text = _lookup_doc_text(Path(meta.path), doc_id) if meta.path else ""
            doc_texts.append((doc_id, text or doc_id))

        sem_idx = build_semantic_index(
            doc_texts,
            model_name=model_name,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        saved_path = sem_idx.save(output)
        err_console.print(
            f"Embedded [bold]{idx.num_docs}[/bold] documents into "
            f"[bold]{len(sem_idx.chunk_doc_ids)}[/bold] chunks "
            f"(shape={sem_idx.embeddings.shape}) -> [cyan]{saved_path}[/cyan]"
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _fail(f"Embedding build failed: {exc}")


@app.command("search")
def cmd_search(
    index_path: Annotated[
        Path,
        typer.Argument(help="Path to the serialized index JSON file."),
    ],
    query: Annotated[
        str,
        typer.Argument(help="Search query string."),
    ],
    top_k: Annotated[
        int,
        typer.Option("-k", "--top-k", min=1, help="Maximum number of results to display."),
    ] = 10,
    scorer: Annotated[
        ScorerName,
        typer.Option("--scorer", help="Ranking model: bm25 or tfidf."),
    ] = "bm25",
    mode: Annotated[
        SearchMode,
        typer.Option("--mode", help="Search mode: lexical, semantic, or hybrid."),
    ] = "lexical",
    embeddings_path: Annotated[
        Path | None,
        typer.Option("--embeddings", "-e", help="Path to .npy embeddings for semantic/hybrid."),
    ] = None,
    boolean: Annotated[
        bool,
        typer.Option("--boolean", help="Parse query using boolean/phrase AST syntax."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit search results as JSON to stdout."),
    ] = False,
) -> None:
    """Search an index in lexical, semantic, or hybrid mode and output Rich table or JSON."""
    try:
        if not query.strip():
            raise ValueError("Query must not be empty")

        idx = load_index(index_path)
        if mode == "lexical":
            if boolean:
                results = search_ast(idx, query, scorer=scorer, top_k=top_k)
            else:
                results = search_ranked(idx, query, scorer=scorer, top_k=top_k)
        else:
            emb_file = embeddings_path or index_path.with_name("embeddings.npy")
            sem_idx = SemanticIndex.load(emb_file)
            if mode == "semantic":
                results = search_semantic(sem_idx, query, top_k=top_k)
            else:
                results = search_hybrid(idx, sem_idx, query, top_k=top_k)

        query_terms = [t for t in tokenize(query) if t not in {"and", "or", "not"}]

        rows: list[tuple[int, str, float, str]] = []
        for rank, res in enumerate(results, start=1):
            snippet = ""
            meta = idx.doc_meta.get(res.doc_id)
            if meta is not None and meta.path is not None:
                text = _lookup_doc_text(Path(meta.path), res.doc_id)
                if text:
                    snippet = generate_snippet(text, query_terms, window=50)
            rows.append((rank, res.doc_id, res.score, snippet))

        if json_output:
            payload = [
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(score, 6),
                    "snippet": snippet,
                }
                for rank, doc_id, score, snippet in rows
            ]
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        out_console = Console()
        if not rows:
            out_console.print(f"No matches found for query: [bold]{query}[/bold]")
            return

        label = f"{mode.upper()} / {scorer.upper()}" if mode == "lexical" else mode.upper()
        table = Table(title=f"Search Results ({label}): {query}")
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Doc ID", style="green")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Snippet", style="white")

        for rank, doc_id, score, snippet in rows:
            table.add_row(str(rank), doc_id, f"{score:.4f}", snippet or "-")

        out_console.print(table)
    except (FileNotFoundError, ValueError, OSError) as exc:
        _fail(str(exc))


@app.command("stats")
def cmd_stats(
    index_path: Annotated[
        Path,
        typer.Argument(help="Path to the serialized index JSON file."),
    ],
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Number of most frequent terms to display."),
    ] = 10,
) -> None:
    """Display summary statistics for a serialized index."""
    try:
        idx = load_index(index_path)
        stats = compute_stats(idx, top_n=top)

        out_console = Console()
        summary = Table(title=f"Index Statistics: {index_path.name}")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right", style="green")
        summary.add_row("Documents", str(stats.num_docs))
        summary.add_row("Unique Terms", str(stats.num_terms))
        summary.add_row("Total Tokens", str(stats.total_tokens))
        summary.add_row("Avg Document Length", f"{stats.avg_doc_length:.2f}")
        out_console.print(summary)

        if stats.top_terms:
            terms_table = Table(title=f"Top {len(stats.top_terms)} Terms")
            terms_table.add_column("Term", style="yellow")
            terms_table.add_column("Total Frequency", justify="right", style="magenta")
            for term, freq in stats.top_terms:
                terms_table.add_row(term, str(freq))
            out_console.print(terms_table)
    except (FileNotFoundError, ValueError, OSError) as exc:
        _fail(str(exc))


if __name__ == "__main__":
    app()
