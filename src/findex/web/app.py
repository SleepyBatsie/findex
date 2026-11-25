"""FastAPI application with lifespan index loading, JSON API, and Jinja2 web UI."""

import json
import logging
import math
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.index import InvertedIndex
from findex.ranking import ScorerName
from findex.search import search_ranked
from findex.stats import compute_stats
from findex.storage import load_index
from findex.tokenize import tokenize
from findex.utils import generate_snippet
from findex.web.config import Settings, get_settings
from findex.web.schemas import (
    DocumentOut,
    HealthResponse,
    SearchHitOut,
    SearchResponse,
    StatsResponse,
    TermFreqOut,
)

logger = logging.getLogger("findex.web")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _read_doc_text(doc_path_str: str | None, doc_id: str) -> str:
    """Read document text from plain file or JSONL line."""
    if not doc_path_str:
        return ""
    file_path = Path(doc_path_str)
    if not file_path.is_file():
        return ""
    if file_path.suffix.lower() != ".jsonl":
        return file_path.read_text(encoding="utf-8", errors="replace")
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = cast(dict[str, object], json.loads(line))
            if str(obj.get("doc_id") or obj.get("url") or "") == doc_id:
                return str(obj.get("text", ""))
    return ""


def _highlight_snippet(snippet: str, query_terms: list[str]) -> str:
    """Wrap matched query tokens in <mark> tags for safe HTML highlighting."""
    import html
    import re

    escaped = html.escape(snippet)
    unique_terms = sorted({t for t in query_terms if t}, key=len, reverse=True)
    if not unique_terms:
        return escaped
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in unique_terms) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


def get_optional_index(request: Request) -> InvertedIndex | None:
    """Dependency returning loaded InvertedIndex or None if unavailable."""
    return cast(InvertedIndex | None, getattr(request.app.state, "index", None))


def require_index(
    idx: Annotated[InvertedIndex | None, Depends(get_optional_index)],
) -> InvertedIndex:
    """Dependency requiring a loaded InvertedIndex (returns HTTP 503 if not ready)."""
    if idx is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded yet.")
    return idx


def _execute_paginated_search(
    idx: InvertedIndex,
    q: str,
    k: int,
    scorer: ScorerName,
    page: int,
) -> SearchResponse:
    """Run ranked search using core dataclasses and slice requested page."""
    all_hits = search_ranked(idx, q, scorer=scorer, top_k=max(idx.num_docs, page * k))
    total_hits = len(all_hits)
    total_pages = math.ceil(total_hits / k) if total_hits > 0 else 0

    start = (page - 1) * k
    end = start + k
    page_slice = all_hits[start:end]

    query_terms = tokenize(q)
    items: list[SearchHitOut] = []
    for offset, hit in enumerate(page_slice):
        meta = idx.doc_meta.get(hit.doc_id)
        raw_text = _read_doc_text(meta.path if meta is not None else None, hit.doc_id)
        snippet = generate_snippet(raw_text, query_terms, window=60) if raw_text else ""
        items.append(
            SearchHitOut(
                rank=start + offset + 1,
                doc_id=hit.doc_id,
                score=round(hit.score, 6),
                snippet=snippet,
            )
        )

    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        page_size=k,
        total_hits=total_hits,
        total_pages=total_pages,
        has_prev=page > 1 and total_pages > 0,
        has_next=page < total_pages,
        results=items,
    )


class _AppLifespan:
    """Async context manager loading the search index once on startup."""

    def __init__(self, application: FastAPI, settings_override: Settings | None = None) -> None:
        self._application = application
        self._settings_override = settings_override

    async def __aenter__(self) -> None:
        cfg = self._settings_override if self._settings_override is not None else get_settings()
        self._application.state.settings = cfg
        self._application.state.index = None
        if cfg.index_path.is_file():
            try:
                loaded = load_index(cfg.index_path)
                self._application.state.index = loaded
                logger.info(
                    "Loaded index from %s (%d docs, %d terms)",
                    cfg.index_path,
                    loaded.num_docs,
                    len(loaded),
                )
            except Exception:
                logger.exception("Failed to load index from %s", cfg.index_path)
        else:
            logger.warning("Index file %s not found; /health will return 503", cfg.index_path)

    async def __aexit__(self, *args: object) -> None:
        self._application.state.index = None


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Application factory creating configured FastAPI instance with lifespan."""
    application = FastAPI(
        title="findex Web Search",
        version="0.7.0",
        lifespan=lambda app_inst: _AppLifespan(app_inst, settings_override),
    )

    @application.middleware("http")
    async def request_id_and_error_boundary(
        request: Request,
        call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception:
            logger.exception(
                "Unhandled exception while processing request_id=%s %s %s",
                req_id,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal Server Error",
                    "request_id": req_id,
                },
                headers={"X-Request-ID": req_id},
            )

    @application.get("/health", response_model=HealthResponse)
    def health_check(
        idx: Annotated[InvertedIndex | None, Depends(get_optional_index)],
    ) -> JSONResponse | HealthResponse:
        if idx is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unavailable",
                    index_loaded=False,
                    num_docs=0,
                    num_terms=0,
                ).model_dump(),
            )
        return HealthResponse(
            status="ok",
            index_loaded=True,
            num_docs=idx.num_docs,
            num_terms=len(idx),
        )

    @application.get("/search", response_model=SearchResponse)
    async def api_search(
        idx: Annotated[InvertedIndex, Depends(require_index)],
        q: Annotated[str, Query(min_length=1, max_length=500, description="Search query")],
        k: Annotated[int, Query(ge=1, le=50, description="Results per page")] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Ranking model")] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000, description="1-based page number")] = 1,
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="Query parameter 'q' must not be blank.")
        return _execute_paginated_search(idx=idx, q=q, k=k, scorer=scorer, page=page)

    @application.get("/docs/{doc_id:path}", response_model=DocumentOut)
    def api_get_document(
        doc_id: str,
        idx: Annotated[InvertedIndex, Depends(require_index)],
    ) -> DocumentOut:
        meta = idx.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        text = _read_doc_text(meta.path, doc_id)
        return DocumentOut(
            doc_id=meta.doc_id,
            length=meta.length,
            path=meta.path,
            text=text,
        )

    @application.get("/stats", response_model=StatsResponse)
    def api_stats(
        idx: Annotated[InvertedIndex, Depends(require_index)],
        top: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> StatsResponse:
        st = compute_stats(idx, top_n=top)
        return StatsResponse(
            num_docs=st.num_docs,
            num_terms=st.num_terms,
            total_tokens=st.total_tokens,
            avg_doc_length=round(st.avg_doc_length, 4),
            top_terms=[TermFreqOut(term=term, frequency=freq) for term, freq in st.top_terms],
        )

    @application.get("/", response_class=HTMLResponse)
    def web_ui_home(
        request: Request,
        idx: Annotated[InvertedIndex | None, Depends(get_optional_index)],
        q: Annotated[str, Query(max_length=500)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        doc: Annotated[str | None, Query()] = None,
    ) -> HTMLResponse:
        search_data: SearchResponse | None = None
        highlighted_hits: list[dict[str, Any]] = []
        selected_doc: DocumentOut | None = None

        if idx is not None and doc:
            meta = idx.doc_meta.get(doc)
            if meta is not None:
                selected_doc = DocumentOut(
                    doc_id=meta.doc_id,
                    length=meta.length,
                    path=meta.path,
                    text=_read_doc_text(meta.path, doc),
                )

        if idx is not None and q.strip():
            search_data = _execute_paginated_search(idx=idx, q=q, k=k, scorer=scorer, page=page)
            q_terms = tokenize(q)
            for item in search_data.results:
                highlighted_hits.append(
                    {
                        "rank": item.rank,
                        "doc_id": item.doc_id,
                        "score": item.score,
                        "snippet_html": _highlight_snippet(item.snippet, q_terms),
                    }
                )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "index_loaded": idx is not None,
                "num_docs": idx.num_docs if idx is not None else 0,
                "num_terms": len(idx) if idx is not None else 0,
                "q": q,
                "k": k,
                "scorer": scorer,
                "page": page,
                "search_data": search_data,
                "highlighted_hits": highlighted_hits,
                "selected_doc": selected_doc,
            },
        )

    return application


app = create_app()