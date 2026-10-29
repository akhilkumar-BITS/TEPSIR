# TEPSIR — Travel and Expense Policy Search and Information Retrieval

Backend service for an M.Tech dissertation project studying retrieval over travel and
expense (T&E) policy documents. Administrators publish policy documents; every other
user searches them in natural language and receives the most relevant policy passages
with their source document and page.

This repository contains the API, the retrieval pipeline and the local infrastructure
definitions. The web client lives in the companion `TEPSIR-ui` repository.

---

## 1. Problem and scope

T&E policy documents are long, cross-referenced and written for auditors as much as for
travellers. Keyword search over such documents performs poorly because a traveller asks
"can I stay an extra night if the meeting runs late?" while the policy states
"additional accommodation nights require prior written approval from the cost centre
owner".

TEPSIR applies retrieval-augmented search:

1. Policy documents are uploaded once by an administrator.
2. The text is extracted, split into overlapping passages and embedded.
3. A user question is embedded into the same vector space.
4. The nearest passages are returned with provenance (document, page, similarity).
5. Optionally, an LLM writes a short answer from those passages and cites them.

Retrieval remains the default and the audit floor: passage results keep every claim
traceable to a source. `POST /search/answer` adds answer synthesis on top — the model
may only use the retrieved passages and cites them by number — while the passages are
always returned alongside the answer. Synthesis degrades to passages-only when the
provider is unavailable or no API key is configured. Evaluation of answer quality is
still on the roadmap.

## 2. Architecture

```
                    ┌────────────────────────────┐
                    │  TEPSIR-ui (React + Vite)  │
                    └─────────────┬──────────────┘
                                  │ HTTPS / JSON
                    ┌─────────────▼──────────────┐
                    │  FastAPI  /api/v1          │
                    │  auth · users · documents  │
                    │  search · answers          │
                    └─────────────┬──────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
        ┌───────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
        │  Services    │  │  Repositories │  │  RAG layer  │
        │  business    │  │  SQL access   │  │  extract    │
        │  rules       │  │               │  │  chunk      │
        └───────┬──────┘  └───────┬───────┘  │  embed      │
                │                 │          │  vector     │
                └─────────────────┼──────────┴──────┬──────┘
                                  │                 │
                   ┌──────────────▼───┐   ┌─────────▼────────┐
                   │ PostgreSQL +     │   │ Object storage   │
                   │ pgvector         │   │ local disk/MinIO │
                   └──────────────────┘   └──────────────────┘
                                  │
                          ┌───────▼───────┐
                          │ Redis (cache) │
                          └───────────────┘
```

Layering rules:

- `api/` only translates HTTP to service calls. No SQL, no business rules.
- `services/` own transactions and policy decisions (who may do what, what happens on
  failure).
- `repositories/` own SQLAlchemy queries.
- `rag/` is storage-agnostic: extraction, chunking, embedding, a `VectorStore`
  protocol, prompt assembly and the optional LLM client. Swapping pgvector for a
  dedicated vector database means implementing one class.
- `storage/` exposes an `ObjectStore` protocol with local-disk and MinIO adapters.

### Indexing flow

```
upload → validate type/size/duplicate
       → store original object (content-addressed by SHA-256)
       → extract text per page (PDF / DOCX / TXT / MD)
       → chunk (≈900 chars, 120 overlap, paragraph-aware)
       → embed chunks (batch)
       → insert documents + document_chunks rows in one transaction
       → mark INDEXED (or FAILED with an error message)
```

Indexing is synchronous within the request so the administrator immediately sees
`INDEXED` or `FAILED`. Moving it to a background worker is a scaling change, not a
contract change.

### Search flow

```
query → normalize/validate → cache lookup (Redis, query embedding only)
      → embed → pgvector cosine search over INDEXED documents
      → top-k passages → log the query → return results with scores
```

Cosine similarity is reported as `1 - cosine_distance`, so 1 is a perfect match.

### Answer flow

```
query → search flow above (passages + one search_queries row logged)
      → number passages in rank order, truncate to the context budget
      → call the LLM with a citation-only system prompt
      → validate [n] markers against the passages actually sent
      → return answer + citations + every retrieved passage
```

Synthesis is optional at every step. A missing key, a timeout, a provider error or an
unusable response returns HTTP 200 with `answer: null`, the passages, and a `warning`
explaining what happened. Markers the model invents are dropped rather than turned into
citations; the full passage list is always present so a reader can verify the answer.

## 3. Roles

| Capability                   | ROLE_ADMIN | ROLE_USER |
| ---------------------------- | :--------: | :-------: |
| Register / sign in           |     ✓      |     ✓     |
| Search policy passages       |     ✓      |     ✓     |
| View own search history      |     ✓      |     ✓     |
| View all users' searches     |     ✓      |           |
| List documents and chunks    |     ✓      |  list only |
| Upload / delete documents    |     ✓      |           |
| List users                   |     ✓      |           |

Self-service registration always creates `ROLE_USER`. The bootstrap administrator is
created from `SEED_ADMIN_EMAIL` / `SEED_ADMIN_PASSWORD` at startup and never through the
public endpoint.

## 4. Repository layout

```
TEPSIR/
├── app/
│   ├── api/            # routers, dependencies, health
│   ├── core/           # settings, security, logging, errors, cache
│   ├── db/             # declarative base, models, async session
│   ├── rag/            # extract, chunk, embed, vector store, prompt, LLM client
│   ├── repositories/   # SQL queries
│   ├── schemas/        # request/response models
│   ├── services/       # auth, documents, search, answer synthesis
│   ├── storage/        # object store adapters
│   ├── main.py         # app factory and lifespan
│   └── seed.py         # bootstrap administrator CLI
├── alembic/            # migrations (0001 = initial schema)
├── docker/             # docker-compose for external resources
├── tests/              # unit tests
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## 5. Running locally

### 5.1 Infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

Starts PostgreSQL with pgvector (5432) and Redis (6379). Documents are stored on local
disk by default, so no object-store service is required. The API itself can also run in
a container:

```bash
docker compose -f docker/docker-compose.yml --profile full up -d --build
```

MinIO is available behind an optional profile (port 9000, console 9001). Public
registries do not always serve the MinIO images anonymously, so start it only if you can
pull them:

```bash
docker compose -f docker/docker-compose.yml --profile minio up -d
# then set STORAGE_BACKEND=minio and the MINIO_* variables in .env
```

### 5.2 API

```bash
cp .env.example .env          # then edit SECRET_KEY
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

alembic upgrade head
uvicorn app.main:app --reload
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/healthz
- Seeded administrator: `admin@example.com` / `admin-password` (change these)

For real embeddings instead of the built-in development embedder:

```bash
pip install -e '.[local-embeddings]'
# .env
EMBEDDING_PROVIDER=sentence-transformers
```

The model must produce `EMBEDDING_DIMENSION`-dimensional vectors (384 for the default
`all-MiniLM-L6-v2`). Changing the dimension requires a migration for the
`document_chunks.embedding` column and re-indexing all documents.

To enable answer synthesis, set a DeepSeek key in `.env`:

```bash
# .env
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-flash
```

Without the key the API still starts and `POST /search/answer` returns passages with
`answer: null` and a warning, so the key is optional for development.

### 5.3 UI

```bash
cd ../TEPSIR-ui
npm install
npm run dev     # http://localhost:5173, proxies /api to :8000
```

## 6. Configuration

All settings are environment variables; defaults live in `app/core/config.py`.

| Variable                    | Default                                                        | Purpose                              |
| --------------------------- | -------------------------------------------------------------- | ------------------------------------ |
| `ENVIRONMENT`               | `development`                                                  | `development`/`test`/`production`    |
| `SECRET_KEY`                | dev placeholder                                                | JWT signing; required in production  |
| `POSTGRES_DSN`              | `postgresql+asyncpg://tepsir:tepsir@localhost:5432/tepsir`      | Database                             |
| `REDIS_URL`                 | `redis://localhost:6379/0`                                     | Query-embedding cache                |
| `SEARCH_CACHE_ENABLED`      | `true`                                                         | Disable cache without removing Redis |
| `STORAGE_BACKEND`           | `local`                                                        | `local` or `minio`                   |
| `EMBEDDING_PROVIDER`        | `dev`                                                          | `dev` or `sentence-transformers`     |
| `EMBEDDING_DIMENSION`       | `384`                                                          | Must match the DB column             |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `120`                                               | Passage granularity                  |
| `MAX_UPLOAD_BYTES`          | `26214400`                                                     | Upload ceiling                       |
| `LLM_ENABLED`               | `true`                                                         | Master switch for answer synthesis   |
| `DEEPSEEK_API_KEY`          | unset                                                          | Enables `/search/answer`; unset = passages only |
| `DEEPSEEK_BASE_URL`         | `https://api.deepseek.com/v1`                                  | OpenAI-compatible endpoint           |
| `LLM_MODEL`                 | `deepseek-flash`                                               | Model identifier sent to the API     |
| `LLM_TIMEOUT_SECONDS`       | `30`                                                           | Read timeout for one completion      |
| `LLM_MAX_OUTPUT_TOKENS`     | `700`                                                          | Answer length ceiling                |
| `LLM_MAX_CONTEXT_CHARS`     | `12000`                                                        | Passage budget per prompt            |
| `LLM_MAX_PASSAGE_CHARS`     | `2000`                                                         | Per-passage truncation ceiling       |
| `SEED_ADMIN_EMAIL` / `SEED_ADMIN_PASSWORD` | unset                                           | Bootstrap administrator              |

`EMBEDDING_PROVIDER=dev` uses a deterministic hashing bag-of-words embedder. It needs no
model download and makes the full stack testable, but it captures lexical overlap only.
Do not report retrieval quality numbers from it; switch to a real model for evaluation.

Answer synthesis sends the user's question and the retrieved policy passages to the
configured third-party API. Do not enable it for content that may not leave the
deployment's network. Prompts, passages, answers and the key are never written to logs;
only status codes, counts and timings are.

## 7. API

Base path: `/api/v1`. Interactive documentation at `/docs`.

| Method | Path                         | Role  | Description                          |
| ------ | ---------------------------- | ----- | ------------------------------------ |
| POST   | `/auth/register`             | —     | Create a `ROLE_USER` account         |
| POST   | `/auth/login`                | —     | Email/password login, returns JWT    |
| GET    | `/users/me`                  | any   | Current profile                      |
| GET    | `/users`                     | admin | Paginated user list                  |
| POST   | `/documents`                 | admin | Upload + index (multipart)           |
| GET    | `/documents`                 | any   | List documents                       |
| GET    | `/documents/{id}`            | any   | Document metadata                    |
| GET    | `/documents/{id}/chunks`     | admin | Indexed passages for inspection      |
| DELETE | `/documents/{id}`            | admin | Delete document, passages and object |
| POST   | `/search`                    | any   | Semantic search                      |
| POST   | `/search/answer`             | any   | Answer from passages, with citations |
| GET    | `/search/history`            | any   | Own history; all history for admins  |

Errors are returned as `{"detail": "...", "code": "..."}` with meaningful status codes
(401, 403, 404, 409, 415, 422).

`POST /search/answer` takes the same body as `/search` and returns:

```json
{
  "query": "Are receipts required for all expenses?",
  "answer": "Yes. The policy states that all expenses require a receipt [1]. ...",
  "citations": [
    {
      "marker": 1,
      "chunk_id": "…",
      "document_id": "…",
      "document_title": "ADP travel policy",
      "page_number": 1,
      "score": 0.2792
    }
  ],
  "results": [ { "chunk_id": "…", "content": "…", "score": 0.2792 } ],
  "model": "deepseek-flash",
  "elapsed_ms": 1020,
  "warning": null
}
```

When synthesis is unavailable the response keeps `results`, sets `answer` and `model`
to `null`, and explains why in `warning`.

## 8. Data model

- `users` — identity and role.
- `documents` — title, metadata, storage key, SHA-256 checksum, indexing status.
  Checksums make duplicate uploads a `409`.
- `document_chunks` — passage text, page number, and a `vector(384)` embedding with an
  HNSW cosine index. Deleting a document cascades to its chunks.
- `search_queries` — audit trail of what was asked, by whom, and how many passages were
  returned. This doubles as the raw material for retrieval evaluation.

## 9. Tests

```bash
pytest
ruff check app tests
```

Unit tests cover chunking, password/JWT handling, the development embedder, prompt
assembly and citation parsing, the LLM client's retry/parse behaviour (with
`httpx.MockTransport`) and the answer service's degrade paths. The end-to-end path
(upload → index → search → answer) is exercised against the Docker stack.

## 10. Design decisions

- **pgvector rather than a separate vector database.** One datastore keeps deletes and
  re-indexing transactional, and HNSW is sufficient for the corpus sizes in scope. The
  `VectorStore` protocol isolates the choice.
- **Passages always, answers optionally.** Every result is attributable to a document
  and page, which is the audit property that motivates the project. A synthesized answer
  is additive; when the provider fails it disappears and the passages remain.
- **Validated citations.** Only markers that match passages actually sent in the prompt
  become citations; invented numbers are dropped and reported in `warning`. The residual
  risk — a valid marker attached to the wrong claim — is why answer quality needs its
  own evaluation before it is trusted.
- **Character-based context budget.** Measuring the prompt in characters avoids shipping
  a tokenizer dependency. Omitted and truncated passages are reported to the client.
- **Content-addressed object storage.** The SHA-256 key gives idempotent writes,
  duplicate detection and a stable way to re-index.
- **Optional Redis.** The cache stores only query embeddings. A cache outage degrades
  latency, never correctness.
- **Synchronous indexing.** Simpler failure reporting for an administrator; the upload
  response carries the final status. A queue is the natural next step if documents
  become large or numerous.

## 11. Roadmap

1. Hybrid retrieval (BM25 + vector) with reciprocal rank fusion.
2. Cross-encoder reranking of the top-k passages.
3. ~~Answer synthesis with passage citations~~ — implemented; evaluating answer quality
   and citation faithfulness is still pending.
4. Retrieval evaluation harness: labelled question set, Recall@k, MRR, nDCG.
5. Background indexing workers and re-index endpoint.
6. Document versioning and an effective-date filter.
7. Per-document access scopes if policy content must be segmented by region.
