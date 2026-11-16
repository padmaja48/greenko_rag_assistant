"""FastAPI app for the Energy Operations Document RAG Assistant."""

from pathlib import Path
import random

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")

from backend.evaluate import build_report  # noqa: E402
from backend.rag import LIMITATIONS, pipeline  # noqa: E402
DATA = BASE / "data"
FRONTEND = BASE / "frontend"
TEST_QUESTIONS = DATA / "questions_test.csv"

app = FastAPI(
    title="Energy Operations Document RAG Assistant",
    description=(
        "RAG assistant for renewable-energy operations, safety, and maintenance SOPs "
        "with citations, retrieval evidence, and hallucination checks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str = Field(default="", description="Operational question to ask the assistant")


@app.get("/api/demo")
def demo():
    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "RAG pipeline ready",
        "expected_outcome": (
            "A working RAG-based assistant that answers operational questions from "
            "provided documents with citations, retrieval evidence, and hallucination checks."
        ),
        "documents_indexed": len({chunk.source for chunk in pipeline.chunks}),
        "chunks_indexed": len(pipeline.chunks),
        "retrieval_backend": pipeline.retrieval_backend,
        "embedding_model": pipeline.embedding_model_name
        if pipeline.retrieval_backend == "fastembed"
        else None,
    }


@app.get("/api/random-test-question")
def random_test_question():
    """Return one random question from the expanded test-question dataset."""
    if not TEST_QUESTIONS.exists():
        return {
            "question": "",
            "expected_source": "",
            "source_dataset": "questions_test.csv",
            "error": "questions_test.csv was not found.",
        }

    df = pd.read_csv(TEST_QUESTIONS).dropna(subset=["question", "expected_source"])
    if df.empty:
        return {
            "question": "",
            "expected_source": "",
            "source_dataset": "questions_test.csv",
            "error": "questions_test.csv contains no usable questions.",
        }

    row = df.iloc[random.randrange(len(df))]
    return {
        "question": str(row["question"]),
        "expected_source": str(row["expected_source"]),
        "source_dataset": "questions_test.csv",
        "total_questions": len(df),
    }


@app.post("/api/ask")
def ask(query: Query):
    return pipeline.ask(query.text)


@app.post("/api/run")
def run_case(query: Query):
    """Backward-compatible alias used by the starter frontend."""
    result = pipeline.ask(query.text)
    return {
        "input": query.text,
        "result": result["answer"],
        "citations": result["citations"],
        "retrieved_chunks": result["retrieved_chunks"],
        "hallucination_check": result["hallucination_check"],
        "limitations": result["limitations"],
        "evaluation_hint": (
            "Use /api/evaluate for retrieval accuracy on questions.csv."
        ),
    }


@app.get("/api/evaluate")
def evaluate():
    """Run smoke, expanded, and off-topic evaluation datasets."""
    report = build_report()
    report["limitations"] = LIMITATIONS
    return report


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
