# Demo notes

## Local demo

1. Copy `.env.example` to `.env` and add `GROQ_API_KEY`, or run without a key to demonstrate extractive fallback.
2. Start the server:

   ```powershell
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

3. Open `http://127.0.0.1:8000`.
4. Demonstrate the following questions:

   - `What should be checked for inverter derating?` — grounded answer and `solar_inverter_sop_test.md` citation.
   - `How should gearbox vibration be investigated?` — grounded answer and `wind_turbine_maintenance_test.md` citation.
   - `How do I bake a chocolate cake?` — abstention with no citation.

5. Click **Run evaluation** and show the smoke, expanded, and off-topic sections.

## Evaluation commands

```powershell
python -m backend.evaluate
```

This writes the complete report to `evaluation_results.json`. The report includes the two-question smoke set, the 394-question expanded set, five off-topic questions, and manual review examples.

## Screenshot checklist

Save screenshots under `docs/screenshots/`:

1. `grounded-answer.png` — answer, citation, retrieval evidence, and grounding badge.
2. `off-topic-abstention.png` — unrelated question, abstention message, and no citation.
3. `evaluation-results.png` — expanded evaluation metrics in the UI or JSON output.
4. `railway-deployment.png` — the deployed homepage with the public URL visible.

Do not show API keys, Railway variable values, or personal account information.

## Observed fallback behavior without a Groq key

- Retrieval and citations remain available.
- Supported questions return an extractive answer from the strongest chunk.
- Off-topic questions return the insufficient-evidence response.

## Safety note

The corpus is synthetic and the assistant is not a substitute for approved site procedures or qualified operational review.
