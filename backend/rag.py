"""RAG pipeline: preprocess docs, embed/retrieve, answer with Groq, and check grounding."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

CHUNK_SIZE = 700
CHUNK_OVERLAP_BULLETS = 1
TOP_K = 4
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MIN_RETRIEVAL_SCORE = 0.05
MIN_ANSWER_SCORE = 0.15  # below this, treat as no useful evidence (avoids off-topic TF-IDF noise)
CITATION_RELATIVE_FLOOR = 0.55  # keep sources within 55% of top score
ABSTAIN_ANSWER = (
    "I do not have enough evidence in the operations documents to answer that confidently."
)
ABSTAIN_PHRASES = (
    "do not know",
    "don't know",
    "not enough evidence",
    "insufficient evidence",
    "cannot answer",
    "can't answer",
    "i do not have enough evidence",
)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "is", "are", "was", "were", "be", "been", "being", "with", "from",
    "by", "as", "that", "this", "these", "those", "it", "its", "should", "must",
    "may", "can", "will", "would", "could", "about", "into", "over", "under",
}


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


class RAGPipeline:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.embedding_model = None
        self.embedding_matrix = None
        self.retrieval_backend = "tfidf"
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embedding_error = None
        self.embedding_status = "disabled"
        self._embedding_thread = None
        self._build_index()

    def _load_documents(self) -> list[tuple[str, str]]:
        """Prefer expanded *_test.md docs when present; otherwise use base SOPs."""
        test_paths = sorted(self.data_dir.glob("*_test.md"))
        if test_paths:
            paths = test_paths
        else:
            paths = sorted(
                p
                for p in self.data_dir.glob("*.md")
                if not p.name.endswith("_test.md")
            )

        docs: list[tuple[str, str]] = []
        for path in paths:
            docs.append((path.name, path.read_text(encoding="utf-8").strip()))
        if not docs:
            raise FileNotFoundError(f"No markdown documents found in {self.data_dir}")
        return docs

    def _chunk_text(self, source: str, text: str) -> list[Chunk]:
        """Chunk by bullet/line groups so procedure steps stay together."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        # Single short doc
        joined = "\n".join(lines)
        if len(joined) <= CHUNK_SIZE:
            return [Chunk(source=source, chunk_id=0, text=joined)]

        chunks: list[Chunk] = []
        chunk_id = 0
        start = 0
        while start < len(lines):
            piece_lines: list[str] = []
            size = 0
            i = start
            while i < len(lines):
                candidate = lines[i]
                extra = len(candidate) + (1 if piece_lines else 0)
                if piece_lines and size + extra > CHUNK_SIZE:
                    break
                piece_lines.append(candidate)
                size += extra
                i += 1
                # Always take at least one line
                if not piece_lines:
                    piece_lines.append(candidate)
                    i = start + 1
                    break

            piece = "\n".join(piece_lines).strip()
            if piece:
                chunks.append(Chunk(source=source, chunk_id=chunk_id, text=piece))
                chunk_id += 1

            if i >= len(lines):
                break
            # Overlap one bullet so procedures aren't split harshly
            start = max(start + 1, i - CHUNK_OVERLAP_BULLETS)

        return chunks

    def _try_build_embedding_index(self, corpus: list[str]) -> None:
        """Build the semantic index, leaving TF-IDF active if it fails."""
        self.embedding_status = "loading"
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name=self.embedding_model_name)
            matrix = np.asarray(list(model.embed(corpus)), dtype=np.float32)
            matrix = matrix / np.maximum(
                np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
            )
            self.embedding_model = model
            self.embedding_matrix = matrix
            self.retrieval_backend = "fastembed"
            self.embedding_status = "ready"
        except Exception as exc:  # pragma: no cover - depends on model/network
            self.embedding_error = f"{type(exc).__name__}: {exc}"
            self.embedding_status = "fallback"

    def _build_index(self) -> None:
        self.chunks = []
        for source, text in self._load_documents():
            self.chunks.extend(self._chunk_text(source, text))
        corpus = [c.text for c in self.chunks]

        # Semantic embeddings are the primary retrieval path. TF-IDF remains a
        # deterministic fallback so the demo still works without model access.
        retrieval_backend = os.getenv("RETRIEVAL_BACKEND", "auto").lower()
        if retrieval_backend not in {"auto", "embeddings", "tfidf"}:
            retrieval_backend = "auto"

        self.matrix = self.vectorizer.fit_transform(corpus)
        self.retrieval_backend = "tfidf"

        if retrieval_backend == "embeddings":
            self._try_build_embedding_index(corpus)
        elif retrieval_backend == "auto":
            # Keep API startup responsive while the local model downloads/loads.
            self._embedding_thread = threading.Thread(
                target=self._try_build_embedding_index,
                args=(corpus,),
                name="embedding-index-loader",
                daemon=True,
            )
            self._embedding_thread.start()

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if self.retrieval_backend == "fastembed":
            query_vec = np.asarray(
                list(self.embedding_model.embed([query])), dtype=np.float32
            )[0]
            query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
            scores = np.asarray(self.embedding_matrix @ query_vec).flatten()
        else:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append(
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "score": round(float(score), 4),
                    "text": chunk.text,
                }
            )
        return results

    def _client_and_model(self) -> tuple[OpenAI, str]:
        """Create the Groq OpenAI-compatible client from environment variables."""
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "No Groq API key found. Set GROQ_API_KEY as an environment variable "
                "for LLM answers."
            )
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return (
            OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"),
            model,
        )

    def generate_answer(self, question: str, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved or retrieved[0]["score"] < MIN_ANSWER_SCORE:
            return ABSTAIN_ANSWER

        context_blocks = []
        for i, item in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[{i}] Source: {item['source']} (relevance_rank={i}, score={item['score']})\n{item['text']}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an energy operations document assistant for renewable-energy "
            "sites (solar, wind, hydro, grid). Answer ONLY using the provided context. "
            "Context blocks are ordered by retrieval relevance — block [1] is strongest. "
            "If the question asks for a procedure, checklist, or what should be done, "
            "return the concrete actionable steps from the most relevant evidence. "
            "Do not answer with a vague confirmation/check line when the context already "
            "contains specific procedural steps (for example recording times, reconciling "
            "values, inspections, or escalations). "
            "Prefer the highest-ranked matching procedure over weaker nearby bullets. "
            "If the question is unrelated to the context (for example cooking or general trivia), "
            f"or the context does not answer it, reply exactly: {ABSTAIN_ANSWER} "
            "Be concise and practical. Mention the primary source filename only when answering. "
            "Use plain text only — do not use markdown bold/italic (no ** or * stars)."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give the direct operational answer using the best-matching steps in the context."
        )

        client, model = self._client_and_model()
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) > 2 and w not in STOPWORDS}

    @staticmethod
    def _is_abstain(answer: str) -> bool:
        lowered = (answer or "").lower()
        return any(p in lowered for p in ABSTAIN_PHRASES)

    def hallucination_check(
        self, answer: str, retrieved: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self._is_abstain(answer) or not retrieved or retrieved[0]["score"] < MIN_ANSWER_SCORE:
            return {
                "supported": False,
                "abstained": True,
                "overlap_ratio": 0.0,
                "notes": "Abstained — no supporting operations evidence for this question.",
            }

        context_text = " ".join(item["text"] for item in retrieved)
        answer_tokens = self._tokens(answer)
        context_tokens = self._tokens(context_text)
        if not answer_tokens:
            return {
                "supported": False,
                "abstained": False,
                "overlap_ratio": 0.0,
                "notes": "Empty answer; marked unsupported.",
            }

        overlap = answer_tokens & context_tokens
        ratio = len(overlap) / len(answer_tokens)
        supported = ratio >= 0.25
        return {
            "supported": supported,
            "abstained": False,
            "overlap_ratio": round(ratio, 3),
            "notes": (
                "Answer terms largely overlap retrieved SOP text."
                if supported
                else "Answer contains many terms not found in retrieved evidence; possible hallucination."
            ),
        }

    @staticmethod
    def _select_evidence(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence = [item for item in retrieved if item["score"] >= MIN_ANSWER_SCORE]
        if not evidence:
            return []
        top = evidence[0]["score"]
        floor = max(MIN_ANSWER_SCORE, top * CITATION_RELATIVE_FLOOR)
        strong = [item for item in evidence if item["score"] >= floor]
        return strong or evidence[:1]

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "question": "",
                "answer": "Please provide a question about the operations documents.",
                "citations": [],
                "retrieved_chunks": [],
                "hallucination_check": {
                    "supported": False,
                    "abstained": False,
                    "overlap_ratio": 0.0,
                    "notes": "Empty question.",
                },
                "limitations": LIMITATIONS,
            }

        retrieved = self.retrieve(question)
        evidence = self._select_evidence(retrieved)

        # Weak / off-topic retrieval: abstain cleanly without citing noisy chunks.
        if not evidence:
            check = self.hallucination_check(ABSTAIN_ANSWER, [])
            return {
                "question": question,
                "answer": ABSTAIN_ANSWER,
                "citations": [],
                "retrieved_chunks": [],
                "hallucination_check": check,
                "limitations": LIMITATIONS,
            }

        try:
            answer = self.generate_answer(question, evidence)
        except RuntimeError:
            # Allow retrieval demo without API key; generation requires key.
            answer = (
                "[Extractive fallback — set GROQ_API_KEY for LLM answers]\n"
                f"{evidence[0]['text']}"
            )

        if self._is_abstain(answer):
            answer = ABSTAIN_ANSWER
            citations: list[str] = []
            evidence_out: list[dict[str, Any]] = []
        else:
            citations = []
            for item in evidence:
                if item["source"] not in citations:
                    citations.append(item["source"])
            evidence_out = evidence

        check = self.hallucination_check(answer, evidence_out)
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": evidence_out,
            "hallucination_check": check,
            "limitations": LIMITATIONS,
        }


LIMITATIONS = (
    "Expanded dummy SOP corpus (*_test.md). TF-IDF may still miss heavy paraphrases. "
    "LLM answers still need human review for safety-critical operations. "
    "Not a substitute for live SCADA data or site-specific procedures."
)

# Shared singleton for the API
pipeline = RAGPipeline()
