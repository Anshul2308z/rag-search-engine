# Product Understanding Document (PUD)
## RAG Search Engine — "Webflyx" Movie Discovery Platform

**Document version:** 1.0  
**Prepared:** 2026-10-12  
**Source of truth:** Repository at `/home/anshul/Documents/Learnings/rag-search-engine`  
**Status:** *Reviewed against live codebase — no assumptions were made without supporting evidence*

---

## 15. Executive Summary

> [!IMPORTANT]
> This is a **learning project**, not a production system. Every section of this document reflects that framing.

`rag-search-engine` is a command-line Python application that implements a complete, from-scratch movie search and retrieval-augmented generation (RAG) pipeline. It was built incrementally by a single developer following a structured course, chapter by chapter. The project uses a dataset of **5,000 movies** (titles + plot descriptions) as its corpus.

The product demonstrates five increasingly sophisticated retrieval strategies — keyword (BM25), semantic (vector), hybrid (BM25 + semantic), chunked semantic search, and multimodal image search — and wraps them with LLM-powered query enhancement, result reranking, and answer generation. The fictional streaming service "Webflyx" is used as a contextual framing for all LLM prompts.

**Maturity:** Functionally complete for a learning project. The core pipeline (index build → search → LLM generation) works end-to-end. Several minor but real bugs exist (detailed in §12). There is no web interface, API server, database, authentication, or deployment configuration.

**Most significant limitations:** No automated tests, no web UI, empty `README.md`, top-level `lib/` directory is empty (phantom artifact), and several runtime bugs in edge-case branches.

---

## 1. Product Overview

| Field | Value |
|---|---|
| **Name** | `rag-search-engine` |
| **Version** | 0.1.0 |
| **Type** | CLI-only Python learning project |
| **Corpus** | 5,000 movies with titles and plot descriptions |
| **Fictional context** | "Webflyx" — a movie streaming service |
| **Core problem** | Learn and implement the full information-retrieval stack from TF-IDF to RAG |
| **README** | Empty (`README.md` contains no content) |
| **Development status** | Feature-complete for the course; no planned future work documented |

### Product Vision

The project is a working implementation of a movie-search engine and RAG assistant. The intended value is **educational**: the developer built every layer of the retrieval pipeline by hand (inverted index, BM25 scoring, sentence embeddings, cosine similarity, RRF fusion, LLM prompting) rather than relying on managed services like Elasticsearch or Pinecone.

---

## 2. Problem Statement and Target Users

### Problem Being Solved (Educational Context)

The developer wanted to understand, by building it, how modern search engines and RAG pipelines work internally:

1. How do keyword search engines build and query an inverted index?
2. How does BM25 improve upon raw TF-IDF?
3. How do dense vector embeddings enable semantic search that keyword matching misses?
4. How does hybrid search combine the strengths of both approaches?
5. How can LLMs enhance queries, rerank results, and generate grounded answers?
6. How can images be used as search queries?

### Target User Groups

| User | Need | Primary Interface |
|---|---|---|
| **Developer (sole user)** | Explore and experiment with retrieval techniques | CLI |
| *(Hypothetical)* Webflyx end-user | Find movies on a streaming platform | Not implemented |

### Main Use Cases (Observed)

1. Searching the movie corpus with a natural-language query (all five search modes).
2. Inspecting intermediate scores (TF, IDF, TF-IDF, BM25-TF, BM25-IDF) for learning purposes.
3. Enhancing a query through LLM spell-correction, rewriting, or term expansion.
4. Reranking search results with an LLM or a cross-encoder model.
5. Generating a grounded natural-language answer (RAG) from top search results.
6. Searching using an image as the query input.
7. Evaluating retrieval quality (precision@k, recall@k, F1) against a curated golden dataset.

### Unverified Assumptions

- The 5,000-movie JSON file is assumed to be high quality; no data-cleaning step is implemented.
- The golden dataset (`data/golden_dataset.json`) is hand-authored for 10 query scenarios.
- The LLM (`apodex/apodex-1.1-mini:free` via OpenRouter) is assumed to be available and reliable; no retries or fallbacks beyond the bare `except Exception` block exist.

---

## 3. Core Features

### 3.1 Inverted Index + Keyword Search

| Field | Detail |
|---|---|
| **Purpose** | Build and query a term-based inverted index of all movie titles and descriptions |
| **Intended user** | Developer experimenting with retrieval |
| **Status** | ✅ Fully implemented |
| **CLI** | `cli/keyword_search_cli.py` |
| **Library** | `cli/lib/keyword_search.py` — `InvertedIndex` class |
| **Data** | Serialized to `cache/` as `index.pkl`, `docmap.pkl`, `term_frequencies.pkl`, `doc_lengths.pkl` |

**What it does:**
- Tokenizes text by stripping punctuation and lowercasing.
- Stems tokens with NLTK's `PorterStemmer`.
- Removes 197 English stopwords loaded from `data/stopwords.txt`.
- Builds a `token → set(doc_ids)` index and a `doc_id → Counter(token)` term-frequency map.
- Saves all structures to `cache/` with Python's `pickle`.

**Commands:**

| Subcommand | What it does |
|---|---|
| `build` | Build the inverted index from `data/movies.json` and save to `cache/` |
| `search <query>` | TF-based keyword search (exact token match, then per-token scan) |
| `tf <doc_id> <term>` | Return raw term frequency of a term in a document |
| `idf <term>` | Return IDF score for a term (smoothed log formula) |
| `tfidf <doc_id> <term>` | Return TF × IDF for a term in a document |
| `bm25tf <doc_id> <term>` | Return BM25 saturated TF score |
| `bm25idf <term>` | Return BM25 IDF score |
| `bm25search <query>` | Full BM25 search ranking all documents |

**Known bug in `search` subcommand:** The output prints only the title string, not a ranked result. The BM25-scored path (`bm25search`) is the correct production-quality search path.

**Known bug in `tf` subcommand:** Line 203 does `print(args.term + " appeared " + term_freq + "times ...")` where `term_freq` is an `int`. This is a runtime `TypeError` because Python does not support int-to-string implicit concatenation.

---

### 3.2 Semantic Search (Dense Vector)

| Field | Detail |
|---|---|
| **Purpose** | Embed movies and queries with a sentence-transformer; rank by cosine similarity |
| **Status** | ✅ Fully implemented |
| **CLI** | `cli/semantic_search_cli.py` |
| **Library** | `cli/lib/semantic_search.py` — `SemanticSearch` class |
| **Model** | `all-MiniLM-L6-v2` (384-dimensional embeddings) |
| **Cache** | `cache/movie_embeddings.npy` — shape `(5000, 384)` |

**What it does:**
- Embeds each movie as `"<title>: <description>"` using `SentenceTransformer`.
- Computes cosine similarity between the query embedding and all movie embeddings.
- Returns the top-k results by similarity score.

**Commands:**

| Subcommand | What it does |
|---|---|
| `verify` | Print model name and max sequence length |
| `embed_text <text>` | Embed a text string and show first 3 dimensions and total dimensions |
| `verify_embeddings` | Load/create embeddings and report shape |
| `embed_query <query>` | Embed a query and show dimensions |
| `search <query>` | Semantic similarity search; returns top results |
| `chunk <text>` | Word-based chunking with configurable size and overlap |
| `semantic_chunk <text>` | Sentence-boundary chunking with configurable max sentences per chunk |
| `embed_chunks` | Build and cache chunk-level embeddings for all movies |
| `search_chunked <query>` | Search against sentence-level chunks; returns movie-level results |

**Chunking implementation detail:** `semantic_chunk()` in `cli/lib/semantic_search.py` splits text on sentence boundaries (`.`, `!`, `?` followed by whitespace) and groups sentences into overlapping windows. Default chunk size is 4 sentences, default overlap is 1.

---

### 3.3 Chunked Semantic Search

| Field | Detail |
|---|---|
| **Purpose** | Improve recall for long movie descriptions by indexing at the chunk level |
| **Status** | ✅ Fully implemented |
| **Library** | `cli/lib/semantic_search.py` — `ChunkedSemanticSearch` class |
| **Cache** | `cache/chunk_embeddings.npy` — shape `(72,909, 384)` and `cache/chunk_metadata.json` |

**What it does:**
- Generates 4-sentence, 1-sentence-overlap chunks from every movie's description.
- Encodes all 72,909 chunks into 384-dimensional vectors.
- At query time, computes similarity against all chunk embeddings, groups chunk scores by movie, takes the max chunk score per movie, and returns the top-k ranked movies.

---

### 3.4 Hybrid Search

| Field | Detail |
|---|---|
| **Purpose** | Combine BM25 and chunked-semantic scores for best-of-both-worlds retrieval |
| **Status** | ✅ Fully implemented |
| **CLI** | `cli/hybrid_search_cli.py` |
| **Library** | `cli/lib/hybrid_search.py` — `HybridSearch` class |

**Two fusion strategies:**

**Weighted Search (`weighted-search`):**
- Normalizes BM25 and semantic scores to `[0, 1]` via min-max normalization.
- Combines as: `hybrid_score = α × bm25_score + (1 - α) × semantic_score`
- Default `α = 0.5` (equal weight).

**Reciprocal Rank Fusion (`rrf-search`):**
- Runs BM25 and chunked semantic search independently, each over 500× the requested limit to ensure broad candidate coverage.
- Converts ranks to scores: `rrf_score(rank, k) = 1 / (rank + k)`, default `k = 60`.
- Sums RRF scores per document across both rankers.
- Returns top results sorted by combined RRF score.
- Supports optional `--enhance` (spell correction, rewrite, expand) and `--rerank-method` (individual LLM, batch LLM, cross-encoder).
- Supports optional `--evaluate` flag to LLM-score results.

**Known bug:** The `--evaluate` flag uses `action="store_true"`, so `args.evaluate` is always `True` or `False`, never `None`. The guard `if args.evaluate is not None` (line 188) is always `True`, meaning evaluation always runs regardless of the flag. Correct guard should be `if args.evaluate:`.

---

### 3.5 LLM Query Enhancement

| Field | Detail |
|---|---|
| **Purpose** | Improve query quality before retrieval |
| **Status** | ✅ Implemented |
| **Module** | `cli/llm_integration.py` |
| **LLM** | `apodex/apodex-1.1-mini:free` via OpenRouter API |
| **External dependency** | `API_KEY` environment variable (OpenRouter key) |

**Three enhancement strategies:**

| Mode | Flag | What the LLM does |
|---|---|---|
| Spell correction | `--enhance spell` | Fix typos without rewording |
| Query rewrite | `--enhance rewrite` | Rewrite to a specific, searchable query (≤10 words) |
| Query expand | `--enhance expand` | Append synonyms and related terms |

**Known bug:** `enhancement_result()` in `llm_integration.py` line 44 compares `query != enhance_query` where `enhance_query` is the *function object*, not the enhanced string. This comparison always evaluates to `True` (a string is never equal to a function), so the "enhanced query" display line always prints regardless of whether the query was actually changed.

---

### 3.6 LLM Reranking

| Field | Detail |
|---|---|
| **Purpose** | Reorder search results by LLM-judged relevance |
| **Status** | ✅ Implemented |
| **Module** | `cli/llm_integration.py` |

**Three reranking methods:**

| Method | Flag | Mechanism |
|---|---|---|
| Individual | `--rerank-method individual` | One LLM call per result; scores 0–10; sorts descending |
| Batch | `--rerank-method batch` | One LLM call for all results; returns JSON array of doc IDs in relevance order |
| Cross-encoder | `--rerank-method cross_encoder` | Uses `cross-encoder/ms-marco-TinyBERT-L2-v2` model locally; scores query–document pairs together |

**Known bug (individual reranking):** `rerank_score()` at line 121 builds a prompt using `doc.get("document", "")`. The `Movie` TypedDict has a `"description"` key, not `"document"`. This means the description is always empty in the reranking prompt, so the LLM scores movies based only on their title.

**Known bug (cross-encoder):** Line 151 of `hybrid_search_cli.py` also uses `doc.get('document', '')` instead of `doc.get('description', '')`, meaning the cross-encoder scores are computed with empty document text.

**Rate limiting:** Individual reranking includes `time.sleep(3)` between LLM calls to avoid hitting API rate limits.

---

### 3.7 Retrieval-Augmented Generation (RAG)

| Field | Detail |
|---|---|
| **Purpose** | Generate natural-language answers grounded in search results |
| **Status** | ✅ Implemented |
| **CLI** | `cli/augmented_generation_cli.py` |
| **Module** | `cli/llm_integration.py` — `augmented_generation()`, `query_llm()` |

**Four RAG modes:**

| Subcommand | What it does |
|---|---|
| `rag <query>` | Searches + generates a grounded answer |
| `summarize <query>` | Synthesizes top results into a 3–4 sentence information-dense summary |
| `citations <query>` | Answers with inline citations `[1]`, `[2]` referencing source documents |
| `question <question>` | Conversational, casual Q&A about available movies |

All prompts are framed for "Webflyx, a movie streaming service" to ensure contextually appropriate responses.

---

### 3.8 Multimodal Search (Image → Movies)

| Field | Detail |
|---|---|
| **Purpose** | Search the movie corpus using an image as the query |
| **Status** | ✅ Implemented |
| **CLI** | `cli/multimodal_search_cli.py` |
| **Library** | `cli/lib/multimodal_search.py` — `Multimodal` class |
| **Model** | `clip-ViT-B-32` (CLIP model via `sentence-transformers`) |

**What it does:**
- Encodes all movie texts (`"<title>: <description>"`) using the CLIP model.
- Opens the provided image (RGB), encodes it with the same CLIP model.
- Computes similarity between image embedding and all text embeddings.
- Returns top-5 results by similarity score.

**Note:** The CLIP text embeddings are computed fresh on every run (not cached to disk). This is a significant performance gap: each invocation re-encodes all 5,000 movies.

**Commands:**

| Subcommand | What it does |
|---|---|
| `verify_image_embedding <path>` | Embed an image and print dimensionality |
| `image_search <path>` | Search movies by image similarity |

---

### 3.9 Multimodal Query Rewriting (Image + Text)

| Field | Detail |
|---|---|
| **Purpose** | Use a vision-capable LLM to synthesize an image and text query into an improved text query |
| **Status** | ✅ Implemented |
| **CLI** | `cli/describe_image_cli.py` |

**What it does:**
- Accepts `--image <path>` and `--query <text>`.
- Base64-encodes the image and sends it alongside the text query to `openrouter/free` (vision-capable model).
- The LLM returns a rewritten text query that synthesizes visual and textual context.
- Intended for pipeline use: the rewritten query can then be fed into text-based search.

**Note:** This is the only CLI that uses `openrouter/free` model rather than `apodex/apodex-1.1-mini:free`.

---

### 3.10 Search Quality Evaluation

| Field | Detail |
|---|---|
| **Purpose** | Measure retrieval quality against a curated golden dataset |
| **Status** | ✅ Implemented |
| **CLI** | `cli/evaluation_cli.py` |
| **Data** | `data/golden_dataset.json` — 10 test cases with queries and lists of relevant movie titles |

**What it does:**
- For each test case, runs RRF hybrid search with `k=60` and returns top-`limit` results.
- Computes Precision@k, Recall@k, and F1@k.
- Prints per-query metrics to stdout.

**Known bug:** Line 49 computes `f1 = 2 * (Recall * Precision) / (Precision + Recall)`. If both Precision and Recall are 0 (no relevant results retrieved), this is a **division by zero** runtime error.

---

## 4. User Journeys and Workflows

### Journey 1: First-Time Setup

```
1. Clone repo
2. Create .env with API_KEY=<openrouter-key>
3. uv sync   # install dependencies
4. python cli/keyword_search_cli.py build     # build inverted index cache
5. python cli/semantic_search_cli.py verify_embeddings  # build movie embeddings cache
6. python cli/semantic_search_cli.py embed_chunks       # build chunk embeddings cache
```

> [!NOTE]
> Steps 4–6 must be performed manually. There is no setup script or entrypoint that automates the cache-building process. The `HybridSearch` constructor auto-builds the inverted index if `cache/index.pkl` is missing, but semantic embeddings must be pre-built explicitly.

**Error states:** If `data/movies.json` is absent (it is gitignored), all commands fail with `FileNotFoundError`. The cache directory is pre-populated in this repo so it would work without re-running steps 4–6, but the data source itself is not in version control.

---

### Journey 2: Keyword Search

```
python cli/keyword_search_cli.py bm25search "animated bear adventure"
```

**Flow:**
1. Load `cache/index.pkl`, `cache/docmap.pkl`, `cache/term_frequencies.pkl`, `cache/doc_lengths.pkl`.
2. Tokenize and stem the query.
3. For each unique query token in the index, iterate over matching documents and accumulate BM25 score.
4. Sort by score descending, return top 5.
5. Print `{rank}. ({id}) {title} - Score: {score}`.

---

### Journey 3: Semantic Search

```
python cli/semantic_search_cli.py search "bear in london marmalade"
```

**Flow:**
1. Load `SentenceTransformer("all-MiniLM-L6-v2")`.
2. Load or create `cache/movie_embeddings.npy` (5000 × 384).
3. Embed the query into a 384-dim vector.
4. Compute cosine similarity against all 5,000 movie embeddings.
5. Sort descending, return top 5.
6. Print title and description.

---

### Journey 4: Hybrid RRF Search with Query Enhancement and Reranking

```
python cli/hybrid_search_cli.py rrf-search "bear movie" \
  --enhance rewrite \
  --rerank-method cross_encoder \
  --limit 5
```

**Flow:**
1. Load movies from `data/movies.json`.
2. Instantiate `HybridSearch`: load/build inverted index + chunk embeddings.
3. Send query to LLM → get rewritten query (e.g., "bear nature film wilderness").
4. Print enhancement log.
5. Run BM25 search (top 2,500 candidates) + chunked semantic search (top 2,500 candidates).
6. Apply RRF fusion with `k=60`, yield top 25 (5 × limit) candidates.
7. Build (query, document) pairs for cross-encoder.
8. Score pairs with `cross-encoder/ms-marco-TinyBERT-L2-v2`.
9. Sort by cross-encoder score, trim to top 5.
10. Print results with RRF score and BM25/semantic ranks.

> [!WARNING]
> Cross-encoder pairs at step 7 use `doc.get('document', '')` instead of `doc.get('description', '')`. The description is always empty in the cross-encoder input, degrading reranking quality.

---

### Journey 5: RAG Answer Generation

```
python cli/augmented_generation_cli.py question "what bear movies are on Webflyx?"
```

**Flow:**
1. Load movies.
2. Instantiate `HybridSearch`.
3. Run RRF search with `k=60`, top 5 results.
4. Build a prompt that includes the 5 search results as context.
5. Call LLM (`apodex/apodex-1.1-mini:free`).
6. Print search results and the LLM's conversational answer.

---

### Journey 6: Image-Based Search

```
python cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

**Flow:**
1. Load all 5,000 movies.
2. Instantiate `Multimodal(movies)` — encodes all 5,000 movie texts with CLIP on every call.
3. Load and encode the image with CLIP.
4. Compute similarity between image embedding and all text embeddings.
5. Print top 5 results with similarity score and description.

---

### Journey 7: Evaluation

```
python cli/evaluation_cli.py --limit 5
```

**Flow:**
1. Load `data/golden_dataset.json` (10 test cases).
2. For each test case, run `HybridSearch.rrf_search(query, k=60, limit=5)`.
3. Compare retrieved titles to the golden `relevant_docs` list.
4. Compute and print Precision@5, Recall@5, F1@5 per query.

---

## 5. Application Structure and Navigation

This is a **CLI-only application** with no web interface or API server. Navigation is through subcommand arguments.

### CLI Entry Points

| Script | Invocation | Purpose |
|---|---|---|
| `cli/keyword_search_cli.py` | `python cli/keyword_search_cli.py <cmd>` | Keyword search, TF/IDF/BM25 inspection |
| `cli/semantic_search_cli.py` | `python cli/semantic_search_cli.py <cmd>` | Semantic search, chunking, embedding |
| `cli/hybrid_search_cli.py` | `python cli/hybrid_search_cli.py <cmd>` | Hybrid search with optional enhancement/reranking |
| `cli/augmented_generation_cli.py` | `python cli/augmented_generation_cli.py <cmd>` | RAG answer generation |
| `cli/multimodal_search_cli.py` | `python cli/multimodal_search_cli.py <cmd>` | Image-based movie search |
| `cli/describe_image_cli.py` | `python cli/describe_image_cli.py --image ... --query ...` | Vision LLM query rewriting |
| `cli/evaluation_cli.py` | `python cli/evaluation_cli.py [--limit N]` | Evaluation metrics against golden dataset |

> [!NOTE]
> All CLIs must be invoked from the **project root** (not from inside `cli/`) because relative paths like `data/movies.json` and `cache/` are resolved from the working directory. The `cli/lib/search_utils.py` module constructs absolute paths using `os.path.dirname(__file__)` for data and cache paths, so those work correctly from any location, but some legacy hard-coded paths in `semantic_search.py` (`cache/movie_embeddings.npy`) are still relative.

### Module Hierarchy

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py       # CLI entry point
│   ├── semantic_search_cli.py      # CLI entry point
│   ├── hybrid_search_cli.py        # CLI entry point
│   ├── augmented_generation_cli.py # CLI entry point
│   ├── multimodal_search_cli.py    # CLI entry point
│   ├── describe_image_cli.py       # CLI entry point
│   ├── evaluation_cli.py           # CLI entry point
│   ├── llm_integration.py          # Shared LLM utility module
│   └── lib/
│       ├── search_utils.py         # Shared constants, types, helpers
│       ├── keyword_search.py       # InvertedIndex class
│       ├── semantic_search.py      # SemanticSearch, ChunkedSemanticSearch
│       ├── hybrid_search.py        # HybridSearch class
│       └── multimodal_search.py    # Multimodal class
├── data/
│   ├── movies.json                 # 5,000-movie corpus (gitignored)
│   ├── golden_dataset.json         # 10 evaluation test cases
│   ├── paddington.jpeg             # Sample image for multimodal testing
│   └── stopwords.txt               # 197 English stopwords
├── cache/                          # Generated artifacts (gitignored)
│   ├── index.pkl                   # Inverted index (85,543 tokens)
│   ├── docmap.pkl                  # doc_id → movie object (5,000 entries)
│   ├── term_frequencies.pkl        # doc_id → Counter(token)
│   ├── doc_lengths.pkl             # doc_id → stemmed token count
│   ├── movie_embeddings.npy        # (5000, 384) float32
│   ├── chunk_embeddings.npy        # (72909, 384) float32
│   └── chunk_metadata.json         # chunk → movie mapping
├── lib/                            # Empty directory (unused artifact)
├── docs/
│   └── product_understanding_document.md  # This document
├── mistakes.md                     # Developer learning log of bugs and fixes
├── pyproject.toml                  # Project manifest and dependencies
├── pyrightconfig.json              # Type checker config
├── .python-version                 # Python 3.14
├── .env                            # API key (gitignored)
├── .env.example                    # Template for API key
└── .gitignore                      # Ignores .venv, data/, cache/, .env
```

---

## 6. System Architecture

### Overview

```
┌────────────────────────────────────────────────────────────┐
│                     CLI Entry Points                       │
│  keyword_  semantic_  hybrid_  augmented_  multimodal_     │
│  search_   search_    search_  generation_ search_  eval_  │
│  cli.py    cli.py     cli.py   cli.py      cli.py   cli.py │
└────────────────────┬───────────────────────────────────────┘
                     │ imports
┌────────────────────▼───────────────────────────────────────┐
│                    cli/lib/ — Core Libraries                │
│                                                             │
│  search_utils.py ← shared constants, types, load_movies()  │
│                                                             │
│  keyword_search.py    semantic_search.py    hybrid_search.py│
│  InvertedIndex        SemanticSearch        HybridSearch    │
│  (BM25)               ChunkedSemanticSearch (RRF/Weighted)  │
│                                                             │
│  multimodal_search.py                                       │
│  Multimodal (CLIP)                                          │
└────────────┬──────────────────────────────┬────────────────┘
             │                              │
  ┌──────────▼──────────┐       ┌───────────▼──────────────┐
  │   Local File Cache  │       │  External Services        │
  │   cache/*.pkl       │       │                           │
  │   cache/*.npy       │       │  OpenRouter API           │
  │   cache/*.json      │       │  (LLM: apodex-1.1-mini)   │
  │                     │       │  (Vision: openrouter/free)│
  └──────────┬──────────┘       └──────────────────────────┘
             │
  ┌──────────▼──────────┐
  │    data/ directory  │
  │   movies.json       │
  │   golden_dataset.json│
  │   stopwords.txt     │
  └─────────────────────┘
```

### Frontend Architecture
Not applicable. There is no frontend.

### Backend Architecture
Python 3.14 CLI application. No web server, no API framework. All code runs as a direct Python process invoked from the shell. The `cli/` directory contains the full application. The `cli/lib/` subdirectory contains reusable library modules.

### Database and Data Persistence
There is no database. Persistence is achieved through:
- **Source data:** `data/movies.json` (5,000 movies, JSON)
- **Inverted index cache:** Four pickle files in `cache/`
- **Semantic embedding cache:** Two NumPy `.npy` files in `cache/`
- **Chunk metadata:** `cache/chunk_metadata.json`

All cache files are gitignored. They must be regenerated on a fresh clone.

### API Communication
The application makes outbound HTTP requests to **OpenRouter** (`https://openrouter.ai/api/v1`) using the `openai` Python SDK with a custom `base_url`. Authentication is via `API_KEY` environment variable. This is the **only external API**.

Sentence-transformer models (`all-MiniLM-L6-v2`, `clip-ViT-B-32`, `cross-encoder/ms-marco-TinyBERT-L2-v2`) are downloaded from Hugging Face on first use and cached locally by the `sentence-transformers` library.

### Authentication and Permissions
The only authentication is the `API_KEY` for OpenRouter. It is stored in `.env` and loaded via `python-dotenv`. No user authentication exists.

### External Integrations

| Integration | Library | Purpose |
|---|---|---|
| OpenRouter LLM API | `openai` SDK | Query enhancement, reranking, RAG generation |
| `all-MiniLM-L6-v2` | `sentence-transformers` | 384-dim text embeddings |
| `clip-ViT-B-32` | `sentence-transformers` | Joint image–text embeddings for multimodal search |
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | `sentence-transformers` | Query–document reranking |
| NLTK PorterStemmer | `nltk` | Token stemming for keyword search |
| Pillow (PIL) | `pillow` | Image loading for CLIP encoding |

### Background Jobs / Async Processing
None. All operations are synchronous. Individual LLM reranking includes `time.sleep(3)` between calls for manual rate limiting.

### Deployment and Hosting
None. This is a local development project. There is no deployment configuration, Docker file, CI/CD pipeline, or cloud infrastructure.

---

## 7. Data Model and Business Logic

### Core Entities

#### Movie
```python
class Movie(TypedDict):
    id: int          # sequential integer (1–5000)
    title: str       # movie title string
    description: str # full plot description (can be very long — multi-paragraph)
```
- Source: `data/movies.json` as `{"movies": [...]}`.
- Description field is used for embedding and indexing.
- **No genre, release year, cast, or rating data** is present in the corpus.

#### SearchResult
```python
class SearchResult(TypedDict):
    id: int
    title: str
    document: str     # truncated description (first 100 chars)
    score: float      # relevance score, rounded to 3 decimal places
    metadata: dict    # optional extra fields
```

#### GoldenTestCase / GoldenDataset
```python
class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]   # list of expected movie titles

class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]
```
- 10 test cases covering bears, wizards, zombies, racing, romcoms, and more.

### InvertedIndex Data Structures
| Attribute | Type | Description |
|---|---|---|
| `index` | `dict[str, set[int]]` | token → set of doc IDs |
| `docmap` | `dict[int, Movie]` | doc_id → full movie object |
| `term_frequencies` | `dict[int, Counter]` | doc_id → stemmed token counts |
| `doc_lengths` | `dict[int, int]` | doc_id → count of stemmed tokens |

- 85,543 unique stemmed tokens in the index.
- 5,000 document entries in docmap.

### Key Constants (from `cli/lib/search_utils.py`)

| Constant | Value | Purpose |
|---|---|---|
| `BM25_K1` | 1.5 | BM25 term saturation parameter |
| `BM25_B` | 0.75 | BM25 length normalization strength |
| `RRF_K` | 60 | RRF smoothing constant |
| `DEFAULT_SEARCH_LIMIT` | 5 | Default number of results returned |
| `DEFAULT_CHUNK_SIZE` | 200 | Default word-based chunk size |
| `DEFAULT_CHUNK_OVERLAP` | 1 | Default chunk overlap (words or sentences) |
| `DEFAULT_SEMANTIC_CHUNK_SIZE` | 4 | Default sentences per semantic chunk |
| `SEARCH_MULTIPLIER` | 5 | Defined but not used for its stated purpose (actual code uses `500 * limit`) |

### Business Rules and Validation

1. **BM25 scoring:** IDF formula is `log((N - df + 0.5) / (df + 0.5) + 1)`. TF formula saturates using length normalization.
2. **RRF candidate pool:** Both BM25 and semantic search retrieve `500 × limit` candidates before fusion to maximize recall before fusion.
3. **Chunking validation:** `semantic_chunk()` raises `ValueError` if `size ≤ 0` or `overlap < 0` or `overlap >= size`.
4. **Empty text:** `generate_embedding()` raises `ValueError` for empty/whitespace-only input. Movie descriptions are silently skipped in chunk building if empty after strip.
5. **Score normalization (weighted search):** Min-max normalization; if all scores are equal, all are set to 1.0 to avoid division by zero.
6. **Cache invalidation:** Embedding cache is invalidated only if the number of cached embeddings differs from the number of documents. There is no content-hash or timestamp check.

---

## 8. API and Integration Inventory

### External APIs

#### OpenRouter Chat Completions

| Field | Value |
|---|---|
| **Base URL** | `https://openrouter.ai/api/v1` |
| **SDK** | `openai` Python library (OpenAI-compatible) |
| **Model (text)** | `apodex/apodex-1.1-mini:free` |
| **Model (vision)** | `openrouter/free` |
| **Auth** | `Authorization: Bearer <API_KEY>` via env var |
| **Request shape** | Standard chat completions: `messages=[{"role":"user","content":"..."}]` |
| **Response** | `response.choices[0].message.content` (string) |
| **Error handling** | Bare `except Exception` block; returns empty string on any failure |

**Use cases:** Query enhancement (spell, rewrite, expand), individual result reranking, batch reranking, RAG answer generation, summarization, cited answers, conversational Q&A, image-query rewriting.

#### Hugging Face Model Hub (Implicit)

| Model | Purpose | Download trigger |
|---|---|---|
| `all-MiniLM-L6-v2` | Text embeddings | First call to `SemanticSearch()` |
| `clip-ViT-B-32` | Multimodal embeddings | First call to `Multimodal()` |
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | Cross-encoder reranking | `--rerank-method cross_encoder` |

These are downloaded automatically by `sentence-transformers` and cached in the HuggingFace local cache (`~/.cache/huggingface/`).

### Internal "API" (Module Interface)

There is no HTTP API. Inter-module communication is via Python imports. The key interface boundary is:

```
CLI scripts  →  cli/lib/  →  data/ + cache/ + External APIs
```

---

## 9. UI/UX Principles

This project has no graphical interface. All user interaction is via the terminal.

### Observed Design Patterns

- **Consistent subcommand structure:** All CLIs use `argparse` with a `subparsers` pattern and a catch-all `case _: parser.print_help()` fallback.
- **Result formatting:** Results are printed with 1-based ranking (`{i+1}.`), title, and a truncated description (first 100 characters) or score.
- **Progressive disclosure:** Score breakdowns (BM25 rank, semantic rank, rerank score) are printed when optional flags are used.
- **Debug logging is commented out:** Several `print()` statements for debugging are preserved as comments (e.g., `hybrid_search_cli.py` lines 79–167).
- **Error output:** Most errors are printed to stdout rather than stderr; some swallow exceptions silently.

### Inconsistencies

- `bm25search` returns `{rank}. ({id}) {title} - Score: {X}` but plain `search` returns only a title string.
- `hybrid_search_cli.py` `evaluate` check `is not None` is always true (see §3.4 bug).
- `test_llm.py` has `base_url = "  "` (only whitespace) — this file is a leftover scratch test and would fail at runtime.
- `describe_image_cli.py` uses `model="openrouter/free"` while all other LLM calls use `model="apodex/apodex-1.1-mini:free"`.

---

## 10. Security, Privacy, and Reliability

### Authentication
- The `API_KEY` for OpenRouter is stored in `.env` (gitignored).
- `.env.example` documents the required variable without exposing a real key.
- `query_llm()` raises `RuntimeError` if `API_KEY` is not set.
- **Risk:** If `.env` is accidentally committed, the key is exposed. The `.gitignore` entry has a trailing space (`".env "`) — this may not be reliable on all systems depending on how the git client handles trailing spaces in `.gitignore` patterns.

### Input Validation
- `semantic_chunk()` validates chunk size and overlap parameters.
- `generate_embedding()` rejects empty/whitespace strings.
- `tokenizeTerm()` rejects multi-token inputs.
- No validation of the `--alpha` parameter range (e.g., values outside `[0, 1]` produce nonsensical hybrid scores).
- No validation of image file existence before passing to CLIP; PIL will raise `FileNotFoundError` at runtime.

### Data Privacy
- Movie data is movie descriptions — no personally identifiable information (PII) is processed.
- User queries are sent to OpenRouter if an enhancement or RAG feature is used. No logging or persistence of queries.

### Error Handling
- LLM errors are caught by `except Exception as e: print("Error encountered:", e); return ""`. Silent fallback to empty string can cause downstream issues (e.g., `json.loads("")` in `rerank_batch` would raise `JSONDecodeError`).
- `InvertedIndex.load()` will raise `FileNotFoundError` if cache is absent; caller is expected to call `build()` first.
- Most CLI commands have no graceful error messages for common issues (missing cache, absent data file).

### Reliability Risks

| Risk | Location | Severity |
|---|---|---|
| `json.loads("")` on LLM failure | `rerank_batch()`, `evaluate_results()` | High — crashes the process |
| Division by zero in F1 | `evaluation_cli.py:49` | Medium — crashes eval when precision+recall=0 |
| `TypeError` in `tf` command | `keyword_search_cli.py:203` | Medium — int + string concatenation |
| Empty description in reranking | `llm_integration.py:121`, `hybrid_search_cli.py:151` | Medium — degrades quality silently |
| CLIP re-encoding all 5,000 movies on each run | `multimodal_search.py` | Low (correctness), High (performance) |

---

## 11. Product Scope

### Existing Capabilities
- ✅ Build and persist an inverted index with BM25 scoring
- ✅ Semantic search via sentence-transformer embeddings
- ✅ Chunked semantic search (sentence-level granularity)
- ✅ Hybrid search (weighted and RRF fusion)
- ✅ LLM query enhancement (spell, rewrite, expand)
- ✅ LLM result reranking (individual, batch, cross-encoder)
- ✅ RAG generation (direct, summarize, cite, question)
- ✅ Multimodal image search (CLIP)
- ✅ Vision-LLM image+text query rewriting
- ✅ Search quality evaluation (precision/recall/F1)
- ✅ Diagnostic commands for TF, IDF, TF-IDF, BM25-TF, BM25-IDF

### Incomplete Capabilities
- ⚠️ Cross-encoder and individual LLM reranking send empty document text to the model (wrong key `"document"` instead of `"description"`)
- ⚠️ CLIP multimodal embeddings are not cached — recomputed on each invocation
- ⚠️ `evaluate` flag logic is inverted — evaluation always runs regardless of the flag
- ⚠️ `enhancement_result()` comparison always logs the enhancement message
- ⚠️ `tf` subcommand crashes on type error at output stage
- ⚠️ `evaluation_cli.py` crashes on division-by-zero when no relevant results are found

### Explicitly Planned Capabilities
None documented. The `README.md` is empty. The git commit messages indicate the project followed a structured course; the final commit message ("Multimodal, completed and course completion") suggests the course is complete and no further features are planned.

### Potential Improvements *(Labeled as Suggestions)*

- **Automated setup script** — A single command to build all caches.
- **Cache for CLIP embeddings** — Save multimodal text embeddings to disk to avoid re-encoding 5,000 movies on every image search.
- **Bug fixes** — Fix the five confirmed runtime bugs listed in §12.
- **Unit tests** — No tests exist; `mistakes.md` notes "No repository test files were present to run."
- **README documentation** — The `README.md` is empty.
- **CLI entrypoints in `pyproject.toml`** — No `[project.scripts]` section; CLIs must be run with explicit `python` paths.
- **Streaming LLM responses** — Currently waits for full completion; streaming would improve perceived latency.
- **Configurable model names** — LLM and embedding model names are hard-coded.

### Out-of-Scope (Evidence-Based)
- Web interface or REST API (no web framework dependency, no route definitions)
- User accounts or authentication (no auth library, no user data model)
- Database (no DB dependency — SQLite, PostgreSQL, etc.)
- Production deployment (no Docker, no CI/CD, no cloud config)
- Real-time or streaming ingestion of new movies

---

## 12. Known Issues and Technical Debt

| # | Issue | Location | Type | Severity |
|---|---|---|---|---|
| 1 | `TypeError`: int + string in `tf` command output | `cli/keyword_search_cli.py:203` | Bug | Medium |
| 2 | `enhancement_result()` compares `query != enhance_query` where `enhance_query` is a function object — always prints | `cli/llm_integration.py:44` | Bug | Low |
| 3 | `rerank_score()` uses `doc.get("document","")` but key is `"description"` — sends empty text to LLM | `cli/llm_integration.py:121` | Bug | Medium |
| 4 | Cross-encoder pairs use `doc.get('document','')` but key is `'description'` — empty document text | `cli/hybrid_search_cli.py:151` | Bug | Medium |
| 5 | `--evaluate` flag with `action="store_true"` — `is not None` guard always True; evaluation runs unconditionally | `cli/hybrid_search_cli.py:188` | Bug | Low |
| 6 | Division by zero in F1 calculation when precision and recall are both 0 | `cli/evaluation_cli.py:49` | Bug | Medium |
| 7 | `json.loads("")` crash when LLM returns empty string in `rerank_batch` or `evaluate_results` | `cli/llm_integration.py:163,190` | Bug | High |
| 8 | CLIP text embeddings not cached — 5,000 movies re-encoded on every `image_search` invocation | `cli/lib/multimodal_search.py` | Performance | Medium |
| 9 | `cache/movie_embeddings.npy` path is hard-coded as relative string — breaks if CWD ≠ project root | `cli/lib/semantic_search.py:123` | Bug | Low |
| 10 | `test_llm.py` has `base_url = "  "` (whitespace only) — would fail at runtime | `cli/test_llm.py:14` | Bug | Low (scratch file) |
| 11 | Top-level `lib/` directory is completely empty — appears to be an unused artifact | `lib/` | Tech debt | Low |
| 12 | `README.md` is empty | `README.md` | Documentation | Low |
| 13 | No automated tests of any kind | Repository-wide | Quality | High |
| 14 | `HybridSearch.__init__` only checks for `index.pkl` existence; if other cache files are absent, `load()` fails | `cli/lib/hybrid_search.py:15` | Bug | Low |
| 15 | `.gitignore` entry for `.env` has a trailing space (`".env "`) which may affect matching behavior | `.gitignore:17` | Security risk | Low |
| 16 | `augmented_generation_cli.py` calls `rrf_search(query, 5)` — second argument is `k`, not `limit`; `limit` defaults to 10 (potentially confusing intent) | `cli/augmented_generation_cli.py:44` | Clarity | Low |
| 17 | `--alpha` has no bounds validation; values outside `[0,1]` produce nonsensical scores | `cli/hybrid_search_cli.py:20` | Robustness | Low |

---

## 13. Product Gaps and Open Questions

| # | Gap / Question | Impact |
|---|---|---|
| 1 | **How was `data/movies.json` created?** The file is gitignored. Its source (scraped, licensed, synthetic?) is unknown. | Medium — reproducibility |
| 2 | **What LLM model is `apodex/apodex-1.1-mini:free`?** The commit history mentions OpenRouter but this model identifier is not a standard OpenRouter model name. It may be a custom or experimental model. | Medium — reliability |
| 3 | **Is the `openrouter/free` model used in `describe_image_cli.py` the same API?** The model name differs from other CLIs. It may route to a different (potentially non-deterministic) model each time. | Low |
| 4 | **No entrypoints in `pyproject.toml`.** Should there be `[project.scripts]` entries for the CLIs? | Low — developer experience |
| 5 | **Why does `Chunks()` exist in `semantic_search_cli.py`** when `semantic_chunk()` in `cli/lib/semantic_search.py` does the same thing differently? The two word-chunking and sentence-chunking approaches are separate but live in the same CLI. | Low — code clarity |
| 6 | **Cache invalidation strategy.** The only cache invalidation is count-based (length check). If the movies dataset changes but retains the same count, stale embeddings will be served silently. | Medium |
| 7 | **`SEARCH_MULTIPLIER = 5` constant exists in `search_utils.py`** but the actual code uses `500 * limit` (100× the constant). The constant appears unused for its stated purpose. | Low — code clarity |

---

## 14. Glossary

| Term | Definition |
|---|---|
| **BM25** | Best Match 25 — a probabilistic ranking function used in keyword search that improves TF-IDF by saturating term frequency and normalizing for document length |
| **BM25_K1** | Saturation parameter (1.5). Controls how quickly BM25 TF saturates — higher values allow term frequency to contribute more |
| **BM25_B** | Length normalization parameter (0.75). Controls the influence of document length on scoring — 0 disables length normalization, 1 applies full normalization |
| **Chunk** | A contiguous sub-sequence of text from a document, used to improve semantic search granularity on long texts |
| **CLIP** | Contrastive Language–Image Pre-Training — a model by OpenAI that learns a shared embedding space for images and text, enabling image-to-text similarity |
| **Cosine similarity** | A measure of similarity between two non-zero vectors as the cosine of the angle between them; used to rank documents by closeness to a query embedding |
| **Cross-encoder** | A transformer model that processes a query–document pair together (rather than independently) to produce a relevance score; more accurate but slower than bi-encoders |
| **Dense embedding** | A high-dimensional floating-point vector representation of text (or image) produced by a neural model |
| **Docmap** | The in-memory dictionary mapping document ID integers to full movie objects |
| **Evaluation** | Measuring search quality with Precision@k, Recall@k, and F1@k metrics against a golden dataset |
| **Golden dataset** | A manually curated set of (query, relevant_document_titles) pairs used as ground truth for evaluation |
| **Hybrid search** | Combining results from multiple retrieval strategies (keyword + semantic) to improve precision and recall |
| **IDF** | Inverse Document Frequency — a measure of how rare a term is across the document collection |
| **Inverted index** | A data structure mapping each unique token to the set of document IDs that contain it, enabling fast keyword lookup |
| **OpenRouter** | An API gateway that provides access to multiple LLMs through an OpenAI-compatible interface |
| **Porter Stemmer** | A rule-based algorithm that reduces words to their root form (e.g., "running" → "run") to improve token matching |
| **Precision@k** | The fraction of the top-k retrieved documents that are relevant |
| **RAG** | Retrieval-Augmented Generation — a technique that retrieves relevant documents and provides them as context to an LLM to generate grounded answers |
| **Recall@k** | The fraction of all relevant documents that appear in the top-k retrieved results |
| **Reranking** | A second-pass scoring step applied after initial retrieval to reorder results by a more accurate (but more expensive) relevance signal |
| **RRF** | Reciprocal Rank Fusion — a rank aggregation technique that combines ranked lists from multiple retrieval systems using `1 / (rank + k)` |
| **Semantic search** | Search using dense embeddings to capture meaning similarity, not just keyword overlap |
| **Sentence-transformers** | A Python library providing pre-trained models for generating high-quality sentence embeddings |
| **Stopword** | A common word (e.g., "the", "is", "at") filtered out before indexing or querying because it carries little discriminative information |
| **TF** | Term Frequency — how many times a term appears in a specific document |
| **TF-IDF** | Term Frequency × Inverse Document Frequency — a classic weighting scheme that scores terms by their frequency in a document relative to their rarity across all documents |
| **Webflyx** | Fictional movie streaming service used as framing context in all LLM prompts |
| **uv** | A fast Python package and project manager used here instead of pip/poetry |
