# Deploying to Cloud Run

Two services: the **API** (`Dockerfile`) and the **Streamlit demo**
(`Dockerfile.demo`). The demo calls the API over HTTP; only the API holds the
OpenAI/Anthropic/DB secrets. Both images are built and verified locally already.

Secrets are passed as Cloud Run env vars at deploy time — never baked into an
image (`.dockerignore` excludes `.env`).

## 0. One-time local setup (your part)

```bash
brew install google-cloud-sdk      # the gcloud CLI (not currently installed)
gcloud auth login                  # opens a browser
gcloud projects create sec-rag-eval-<unique>   # or use an existing project
gcloud config set project sec-rag-eval-<unique>
# Enable billing for the project in the console (free tier, but a card is required):
#   https://console.cloud.google.com/billing
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Pick a region once (Neon is us-east-1, so keep latency low):

```bash
gcloud config set run/region us-east1
```

## 1. Choose a shared API key

The API guards `/query` with `SEC_RAG_API_KEY` (an `X-API-Key` header). Generate one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Keep it — both the API and the demo need it.

## 2. Deploy the API

Cloud Run can build from source (no manual docker push needed). From the repo root:

```bash
gcloud run deploy sec-rag-api \
  --source . \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 60 \
  --set-env-vars "OPENAI_API_KEY=sk-...,ANTHROPIC_API_KEY=sk-ant-...,DATABASE_URL=postgresql://...,SEC_RAG_API_KEY=<the key from step 1>"
```

Notes:
- `--source .` uses `Dockerfile` automatically.
- `--allow-unauthenticated` makes the URL reachable; the `SEC_RAG_API_KEY` guard
  is what actually protects `/query` from draining your model keys.
- `--memory 1Gi` because pgvector + the SDKs need headroom; 512Mi can OOM.
- Quote the whole `--set-env-vars` value; commas separate vars, so a value
  containing a comma needs the `^@^`-delimiter form (DB URLs usually don't).

Grab the URL it prints, e.g. `https://sec-rag-api-xxxx.us-east1.run.app`.

Smoke test:

```bash
curl https://sec-rag-api-xxxx.us-east1.run.app/health
curl -X POST https://sec-rag-api-xxxx.us-east1.run.app/query \
  -H "Content-Type: application/json" -H "X-API-Key: <the key>" \
  -d '{"query":"What were Boeing'\''s risk factors in 2022?"}'
```

## 3. Deploy the demo

```bash
gcloud run deploy sec-rag-demo \
  --source . \
  --dockerfile Dockerfile.demo \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars "SEC_RAG_API_URL=https://sec-rag-api-xxxx.us-east1.run.app,SEC_RAG_API_KEY=<the key from step 1>"
```

Open the demo URL it prints — that is the link to share.

## Costs & behaviour

- **Scale to zero:** you pay nothing when idle. Per-query API cost is unchanged
  (~$0.006/query, same as local).
- **Cold start:** first request after idle wakes the container (~10–20 s) on top
  of the ~15 s query time. Warm requests are just the query time.
- **The guard is your wallet's protection.** Without `SEC_RAG_API_KEY`, a public
  `/query` URL would let anyone spend your OpenAI/Anthropic credits.

## Updating

Re-run the same `gcloud run deploy ...` command; Cloud Run rebuilds and rolls out.

## Local container test (already verified)

```bash
docker build -t sec-rag-api .
docker run -p 8090:8080 --env-file .env -e SEC_RAG_API_KEY=test sec-rag-api
curl localhost:8090/health
```
