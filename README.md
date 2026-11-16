# Greenko Energy Operations RAG Assistant

An evidence-grounded document assistant for renewable-energy operations, safety, and maintenance procedures. The application retrieves relevant passages from bundled SOP-style documents, generates a concise answer with Groq, displays citations and retrieval evidence, and abstains when the corpus does not provide enough support.

## Problem framing

Solar, wind, hydro, and grid operations teams need quick access to procedures without relying on an unconstrained language model. This prototype demonstrates a retrieval-augmented generation (RAG) workflow with source attribution and a lightweight grounding check.

## Features

- Markdown document loading, cleaning, and line-aware chunking
- FastEmbed dense embeddings and cosine-similarity retrieval
- Deterministic TF-IDF fallback when embeddings are unavailable
- Groq LLM answer generation through Groq's OpenAI-compatible API
- Extractive fallback when `GROQ_API_KEY` is not configured
- Source citations and retrieved evidence
- Lexical grounding / hallucination signal and low-evidence abstention
- Browser UI served by FastAPI
- Smoke, expanded, and off-topic evaluation reports
- Railway deployment configuration

## Architecture

```text
Markdown SOPs -> preprocess in backend/rag.py -> chunks -> embeddings/cosine retrieval
-> Groq answer constrained to evidence -> citations, grounding check, abstention
```

## Key files

- `backend/app.py` — FastAPI API and static UI serving.
- `backend/rag.py` — preprocessing, chunking, retrieval, generation, and checks.
- `backend/evaluate.py` — evaluation runner and report generation.
- `data/*.md` — compact source documents.
- `data/*_test.md` — expanded synthetic bulk documents for end-to-end testing.
- `data/questions.csv` — two-question smoke-test set.
- `data/questions_test.csv` — 394-question expanded retrieval set.
- `data/off_topic_questions.csv` — unrelated-question abstention set.
- `frontend/index.html` — single-page demonstration UI.
- `docs/screenshots/` — submission screenshot directory.

## Dataset and preprocessing

The bundled Markdown files are synthetic, safety-oriented sample documents created for this case study; they are not live Greenko plant procedures or SCADA data.

Preprocessing is implemented in `backend/rag.py`. At startup, the pipeline:

1. Loads Markdown documents from `data/`.
2. Prefers files ending in `_test.md` when they are available.
3. Removes empty lines and normalizes line whitespace.
4. Groups procedure lines into chunks of approximately 700 characters with one-line overlap.
5. Encodes chunks with FastEmbed using `BAAI/bge-small-en-v1.5` and uses cosine similarity for retrieval.
6. Falls back to a TF-IDF matrix if the embedding package/model is unavailable or `RETRIEVAL_BACKEND=tfidf` is set.

The `*_test.md` files are intentionally expanded bulk versions of the compact source documents. They contain more procedural coverage so that the full retrieval path can be tested end to end. They should be understood as a synthetic test corpus, not as a separate production data source. The `questions_test.csv` file is similarly an expanded, source-labelled evaluation set.

Groq is used for answer generation. Groq's current public API does not expose an embedding endpoint/model, so semantic retrieval uses the local sentence-transformer model above rather than pretending to call a non-existent Groq embedding model.

## Local setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The key is loaded only from the environment. Without a key, retrieval, citations, abstention, and extractive fallback remain available; only generated LLM answers are disabled.

The first run may download the quantized FastEmbed model. To use the lightweight local fallback explicitly, set `RETRIEVAL_BACKEND=tfidf`. No API key is required for embeddings.

Run the application:

```powershell
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000>.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Browser demo |
| GET | `/api/demo` | Health and index statistics |
| POST | `/api/ask` | Ask a question: `{"text":"..."}` |
| POST | `/api/run` | Backward-compatible response alias |
| GET | `/api/evaluate` | Run all evaluation datasets |

## Evaluation

Run the complete local report with:

```powershell
python -m backend.evaluate
```

The generated `evaluation_results.json` separates the results into:

- **Smoke test:** the two questions in `data/questions.csv`.
- **Expanded evaluation:** all 394 questions in `data/questions_test.csv`.
- **Off-topic evaluation:** unrelated questions in `data/off_topic_questions.csv`; the expected behavior is abstention with no citation.
- **Manual review examples:** representative grounded, procedural, and off-topic cases.

Reported metrics include top-1 source accuracy, top-3 source accuracy, mean reciprocal rank, retrieval abstentions, and off-topic abstention accuracy. The evaluation is a synthetic, source-level retrieval benchmark. It is not a production accuracy, faithfulness, or safety guarantee.

Latest local run:

| Evaluation | Top-1 | Top-3 | MRR | Abstentions |
|---|---:|---:|---:|---:|
| Smoke test (2) | 1.000 | 1.000 | 1.000 | 0 |
| Expanded set (394) | 0.891 | 0.977 | 0.933 | 46 |

Off-topic abstention accuracy was `1.000` across 5 questions. These values are reproducible with `python -m backend.evaluate` and are stored in `evaluation_results.json`.

## Design decisions and limitations

| Decision | Choice | Reason |
|---|---|---|
| Retrieval | FastEmbed dense embeddings + cosine similarity | Captures semantic similarity and supports paraphrased questions |
| Fallback retrieval | TF-IDF + cosine similarity | Keeps the demo usable without model download/access |
| Generation | Groq via OpenAI-compatible SDK | Hosted generation with credentials supplied through environment variables |
| Chunking | Line-aware chunks with overlap | Keeps procedure steps together and preserves local context |
| Grounding check | Lexical overlap plus retrieval threshold | Simple, transparent signal for a screening prototype |
| Frontend | Single static HTML page | Minimal deployment surface and clear evidence display |

Known limitations:

- The corpus and evaluation questions are synthetic and small.
- Embedding quality depends on the selected local model; the model is general-purpose rather than energy-domain fine-tuned.
- The TF-IDF fallback can miss paraphrases and semantic synonyms.
- The grounding check is a heuristic, not a formal faithfulness or NLI evaluation.
- There is no authentication, rate limiting, audit logging, PII handling, or live SCADA integration.
- Safety-critical answers require review by qualified site personnel.
- The current application is not a substitute for approved, site-specific procedures.

## Demo and screenshots

See [DEMO_NOTES.md](DEMO_NOTES.md) for the exact demo flow. Before submission, add these files under [docs/screenshots](docs/screenshots/):

- `grounded-answer.png`
- `off-topic-abstention.png`
- `evaluation-results.png`
- `railway-deployment.png`

Do not capture API keys or private account details in screenshots.

## Railway deployment

The repository includes `Procfile`, `railway.toml`, and `nixpacks.toml`. Create a Railway service from this repository and configure:

```text
GROQ_API_KEY=<Railway secret variable>
GROQ_MODEL=openai/gpt-oss-20b
```

The service start command is:

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

After deployment, smoke-test `/`, `/api/demo`, `/api/ask`, and `/api/evaluate`. Record the final public URL below only after verifying it in a private browser window.

**Live URL:** <https://greenko-rag-assistant-production.up.railway.app>

**Latest Railway deployment:** <https://railway.com/project/7381b650-24e7-4239-8735-045c789d02ae/service/e8013f4b-591b-4351-81bd-004a20834a9c?id=e1102828-3308-49f8-ac77-060d2bf67eeb>

See [DEPLOY.md](DEPLOY.md) for deployment and smoke-test steps.

## Submission checklist

- [x] Backend source code
- [x] Frontend source code
- [x] Bundled dataset and evaluation datasets
- [x] Preprocessing and chunking documented in `backend/rag.py`
- [x] Setup and run commands
- [x] Evaluation script and generated results
- [ ] Add four final screenshots under `docs/screenshots/`
- [ ] Verify and add the Railway public URL
- [x] Secrets supplied through environment variables only
