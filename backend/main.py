import json
import queue
import asyncio
import threading
from typing import Generator
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="ABAD Branding & Labeling AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


class AnalyzeRequest(BaseModel):
    product: str


class Phase2Request(BaseModel):
    product: str
    stage1: dict
    stage2: dict
    stage3: dict
    user_colors: list


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {"status": "ok"}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _pipeline_stream(gen: Generator) -> StreamingResponse:
    """Bridge a synchronous pipeline generator to a streaming SSE response."""

    async def event_generator():
        result_queue: queue.Queue[str | None] = queue.Queue()

        def run_in_thread() -> None:
            try:
                for chunk in gen:
                    result_queue.put(chunk)
            except Exception as e:
                result_queue.put(
                    json.dumps(
                        {"stage": "pipeline", "status": "error", "data": {"error": str(e)}},
                        ensure_ascii=False,
                    )
                )
            finally:
                result_queue.put(None)

        threading.Thread(target=run_in_thread, daemon=True).start()

        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, result_queue.get)
            if chunk is None:
                break
            yield f"data: {chunk}\n\n"

        yield _sse({"stage": "done", "status": "complete", "data": {}})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if not request.product or not request.product.strip():
        raise HTTPException(status_code=400, detail="Product input cannot be empty")
    from backend.pipeline import run_pipeline_phase1
    return _pipeline_stream(run_pipeline_phase1(request.product.strip()))


@app.post("/api/analyze/continue")
async def analyze_continue(request: Phase2Request):
    """Phase 2: run stages 4-7 with user-confirmed colours."""
    from backend.pipeline import run_pipeline_phase2
    return _pipeline_stream(
        run_pipeline_phase2(request.stage1, request.stage2, request.stage3, request.user_colors)
    )
