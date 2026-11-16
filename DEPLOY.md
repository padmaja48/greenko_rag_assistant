# Railway deployment

The app is configured for Railway with `Procfile`, `railway.toml`, and `nixpacks.toml`.

## Deploy from the GitHub repository

1. Push the completed repository to GitHub.
2. In Railway, create a new project from the GitHub repository.
3. Add these service variables in Railway. Do not put their values in source control:

   ```text
   GROQ_API_KEY=<secret>
   GROQ_MODEL=openai/gpt-oss-20b
   ```

4. Deploy and generate a public domain.
5. Open the public URL in a private browser window.

The start command is:

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

## Smoke test after deployment

Verify all of the following:

- `/` loads the frontend.
- `/api/demo` reports the indexed document and chunk counts.
- `/api/ask` answers `What should be checked for inverter derating?` and cites `solar_inverter_sop_test.md`.
- `/api/ask` abstains for `How do I bake a chocolate cake?`.
- `/api/evaluate` returns smoke, expanded, and off-topic evaluation sections.

Capture a screenshot for `docs/screenshots/railway-deployment.png`, ensuring no secret values are visible. Add the verified public URL to `README.md` only after these checks pass.

## Railway CLI alternative

```powershell
railway login
railway link
railway up
railway variables set GROQ_API_KEY=<secret>
railway variables set GROQ_MODEL=openai/gpt-oss-20b
railway domain
```

The first request may be slower if the service has been idle. Never commit `.env` or an API key.
