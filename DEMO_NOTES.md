# Demo notes

## Local demo

1. Copy `.env.example` to `.env` and add `GROQ_API_KEY`, or run without a key to demonstrate extractive fallback.
2. Start the server:

   ```powershell
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

3. Open `http://127.0.0.1:8000`.
4. Confirm the active retrieval backend in `/api/demo`. Local `auto` mode attempts FastEmbed using `BAAI/bge-small-en-v1.5`; the live Railway demo intentionally reports `tfidf` because it uses the resource-safe deployment profile documented in `DEPLOY.md`.
5. Demonstrate the following questions:

   - `What should be checked for inverter derating?` — grounded answer and `solar_inverter_sop_test.md` citation.
   - `How should gearbox vibration be investigated?` — grounded answer and `wind_turbine_maintenance_test.md` citation.
   - `How do I bake a chocolate cake?` — abstention with no citation.

6. Run `python -m backend.evaluate` to generate the smoke, expanded, and off-topic report. The same report is available from the retained `GET /api/evaluate` endpoint.

## Evaluation commands

```powershell
python -m backend.evaluate
```

This writes the complete report to `evaluation_results.json`. The report includes the two-question smoke set, the 394-question expanded set, five off-topic questions, and manual review examples.

## Screenshot checklist

Save screenshots under `docs/screenshots/`:

1. `grounded-answer.png` — answer, citation, retrieval evidence, and grounding badge.
2. `off-topic-abstention.png` — unrelated question, abstention message, and no citation.
3. `railway-deployment.jpeg` — the deployed homepage with the public URL visible.

Do not show API keys, Railway variable values, or personal account information.

## Observed fallback behavior without a Groq key

- Retrieval and citations remain available.
- Supported questions return an extractive answer from the strongest chunk.
- Off-topic questions return the insufficient-evidence response.

## Safety note

The corpus is synthetic and the assistant is not a substitute for approved site procedures or qualified operational review.
