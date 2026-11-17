"""Typer + Rich command-line interface for findex."""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.corpus import Document, discover_files, read_document
from findex.index import InvertedIndex
from findex.ranking import ScorerName
from findex.search import search_ast, search_ranked
from findex.stats import compute_stats
from findex.storage import load_index, save_index
from findex.tokenize import tokenize
from findex.utils import generate_snippet

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="Full-text search engine CLI with BM25/TF-IDF ranking and AST queries.",
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


@app.command("index")
def cmd_index(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Directory or file containing documents to index."),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Destination JSON path for the built index."),
    ] = Path("index.json"),
) -> None:
    """Build an inverted index from a directory of documents."""
    err_console = Console(stderr=True)
    try:
        files = discover_files(corpus_dir)
        base_dir = corpus_dir if corpus_dir.is_dir() else corpus_dir.parent

        docs: list[Document] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
        ) as progress:
            task = progress.add_task("Indexing documents...", total=len(files))
            for file_path in files:
                docs.append(read_document(file_path, base_dir=base_dir))
                progress.advance(task)

        idx = InvertedIndex()
        idx.build(docs)
        saved_path = save_index(idx, output)
        logger.info("Index successfully saved to %s", saved_path)
        err_console.print(
            f"Indexed [bold]{idx.num_docs}[/bold] documents "
            f"([bold]{len(idx)}[/bold] unique terms) -> [cyan]{saved_path}[/cyan]"
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        _fail(str(exc))


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
    boolean: Annotated[
        bool,
        typer.Option("--boolean", help="Parse query using boolean/phrase AST syntax."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit search results as JSON to stdout."),
    ] = False,
) -> None:
    """Search an index and output either a Rich table or JSON."""
    try:
        if not query.strip():
            raise ValueError("Query must not be empty")

        idx = load_index(index_path)
        if boolean:
            results = search_ast(idx, query, scorer=scorer, top_k=top_k)
        else:
            results = search_ranked(idx, query, scorer=scorer, top_k=top_k)

        query_terms = [t for t in tokenize(query) if t not in {"and", "or", "not"}]

        rows: list[tuple[int, str, float, str]] = []
        for rank, res in enumerate(results, start=1):
            snippet = ""
            meta = idx.doc_meta.get(res.doc_id)
            if meta is not None and meta.path is not None:
                doc_file = Path(meta.path)
                if doc_file.is_file():
                    text = doc_file.read_text(encoding="utf-8", errors="replace")
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

        table = Table(title=f"Search Results ({scorer.upper()}): {query}")
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Doc ID", style="green")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Snippet", style="white")

        for rank, doc_id, score, snippet in rows:
            table.add_row(str(rank), doc_id, f"{score:.4f}", snippet or "—")

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