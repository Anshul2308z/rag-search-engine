# RAG Search Engine

A modular, from-scratch Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) search engine built in Python.

This project explores the modern search stack progressively—from foundational term-based inverted indexes and BM25 ranking, to dense vector semantics, chunk-level embeddings, hybrid rank fusion (RRF), cross-encoder / LLM reranking, multimodal image search, and grounded RAG answer generation.

All features are framed around a movie discovery assistant for **Webflyx**, an example streaming service indexing a corpus of 5,000 movies.

---

## Table of Contents

- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Configuration](#configuration)
- [Quickstart & Usage](#quickstart--usage)
  - [1. Data Indexing & Cache Setup](#1-data-indexing--cache-setup)
  - [2. Keyword Search (BM25 & TF-IDF)](#2-keyword-search-bm25--tf-idf)
  - [3. Semantic & Chunked Search](#3-semantic--chunked-search)
  - [4. Hybrid Search (Weighted & RRF)](#4-hybrid-search-weighted--rrf)
  - [5. Retrieval-Augmented Generation (RAG)](#5-retrieval-augmented-generation-rag)
  - [6. Multimodal Image Search](#6-multimodal-image-search)
  - [7. Evaluation](#7-evaluation)
- [Evaluation Dataset](#evaluation-dataset)
- [Documentation](#documentation)

---

## Features

- **Keyword Retrieval & Lexical Scoring**:
  - Custom inverted index builder tokenizing text, stripping punctuation, filtering stopwords, and stemming via NLTK's `PorterStemmer`.
  - Analytical tools for TF, smoothed IDF, and TF-IDF calculation.
  - Full BM25 implementation with document length normalization and tunable saturation parameters ($k_1 = 1.5$, $b = 0.75$).

- **Dense Semantic & Chunked Search**:
  - Vector embeddings powered by `all-MiniLM-L6-v2` (384-dimensional) via `sentence-transformers`.
  - Sentence-boundary semantic chunking (sliding window with configurable size & overlap) to capture targeted relevance across long plot summaries.
  - Aggregated max-score ranking across individual document chunks.

- **Hybrid Fusion & Reranking**:
  - Min-Max normalized weighted score fusion: $\alpha \cdot \text{BM25} + (1 - \alpha) \cdot \text{Semantic}$.
  - Reciprocal Rank Fusion (RRF) with configurable smoothing factor $k$ ($k=60$ default).
  - LLM Query Enhancement: automatic spelling correction, Google-style query rewriting, and semantic keyword expansion.
  - Second-pass reranking:
    - **Local Cross-Encoder**: `cross-encoder/ms-marco-TinyBERT-L2-v2`.
    - **LLM Individual**: Scoring query-document pairs on a 0–10 scale.
    - **LLM Batch**: Global ranking returning an ordered list of document IDs.

- **Retrieval-Augmented Generation (RAG)**:
  - Contextual answer generation grounded in top search hits via OpenRouter (`apodex/apodex-1.1-mini:free`).
  - Multi-document plot summarization.
  - Cited query answering with numerical bracketed sources (`[1]`, `[2]`).
  - Conversational streaming assistant for user movie recommendations.

- **Multimodal Search & Vision Processing**:
  - Zero-shot image-to-text movie search using OpenAI's `clip-ViT-B-32`.
  - Multimodal query rewriting combining image inputs and textual queries using vision-capable LLMs.

- **Benchmarking & Evaluation**:
  - Evaluation harness computing **Precision@k**, **Recall@k**, and **F1 Score** against a curated 10-query golden dataset.

---

## Architecture & Tech Stack

- **Language**: Python 3.14+
- **Environment & Package Management**: `uv`
- **Embeddings & Models**:
  - `sentence-transformers` (`all-MiniLM-L6-v2`, `clip-ViT-B-32`, `cross-encoder/ms-marco-TinyBERT-L2-v2`)
  - `torch` (CPU index build)
- **Natural Language Processing**:
  - `nltk` (PorterStemmer)
  - Custom regex sentence segmenters
- **Vector Operations & Caching**:
  - `numpy` vector matrices (`.npy`)
  - Python `pickle` for inverted index serialization
- **LLM & Vision Integration**:
  - `openai` SDK routed to OpenRouter API
  - `pillow` (PIL) for image decoding

---

## Project Structure

```text
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py       # BM25, TF, IDF, inverted index CLI
│   ├── semantic_search_cli.py      # Semantic embeddings & chunked search CLI
│   ├── hybrid_search_cli.py        # Weighted & RRF hybrid search CLI
│   ├── augmented_generation_cli.py # RAG answers, summaries, & citations CLI
│   ├── multimodal_search_cli.py    # CLIP-based image search CLI
│   ├── describe_image_cli.py       # Vision LLM multimodal query rewriter CLI
│   ├── evaluation_cli.py           # Precision@k / Recall@k evaluation harness
│   ├── llm_integration.py          # LLM queries, prompt templates, & rerankers
│   └── lib/
│       ├── keyword_search.py       # InvertedIndex & BM25 scoring logic
│       ├── semantic_search.py      # SemanticSearch & ChunkedSemanticSearch
│       ├── hybrid_search.py        # HybridSearch, RRF & normalization
│       ├── multimodal_search.py    # Multimodal CLIP image search class
│       └── search_utils.py         # Data models, paths, and shared helpers
├── data/
│   ├── movies.json                 # 5,000-movie corpus (titles and descriptions)
│   ├── golden_dataset.json         # 10 labeled test cases for evaluation
│   ├── paddington.jpeg             # Test image for multimodal search
│   └── stopwords.txt               # 197 standard English stopwords
├── cache/                          # Serialized indices and vector embeddings
│   ├── index.pkl
│   ├── docmap.pkl
│   ├── term_frequencies.pkl
│   ├── doc_lengths.pkl
│   ├── movie_embeddings.npy
│   ├── chunk_embeddings.npy
│   └── chunk_metadata.json
├── docs/
│   └── product_understanding_document.md # Comprehensive architectural & product audit
├── pyproject.toml                  # Project manifest and dependencies
├── uv.lock                         # Lockfile
└── README.md
```

---

## Prerequisites & Installation

1. **Install [`uv`](https://github.com/astral-sh/uv)** (recommended Python package runner):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository and install dependencies**:
   ```bash
   git clone <repo-url>
   cd rag-search-engine
   uv sync
   ```

---

## Configuration

To use the LLM-powered features (query enhancement, LLM reranking, and RAG), configure your OpenRouter API key:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set your key:
   ```env
   API_KEY="your-openrouter-api-key"
   ```

*Note: Keyword search, semantic vector search, local cross-encoder reranking, and CLIP multimodal search run locally and do not require an API key.*

---

## Quickstart & Usage

> [!TIP]
> Run all CLI commands from the project root directory.

### 1. Data Indexing & Cache Setup

Before querying, generate the cached inverted index and dense embeddings:

```bash
# Build inverted index (lexical frequencies, doc lengths, doc map)
uv run python cli/keyword_search_cli.py build

# Generate and verify dense document embeddings
uv run python cli/semantic_search_cli.py verify_embeddings

# Generate and cache chunk-level embeddings (for chunked search)
uv run python cli/semantic_search_cli.py embed_chunks
```

---

### 2. Keyword Search (BM25 & TF-IDF)

Execute BM25 searches or inspect term statistics:

```bash
# BM25 search across 5,000 movies
uv run python cli/keyword_search_cli.py bm25search "bear marmalade london"

# Inspect BM25 Inverse Document Frequency (IDF) for a term
uv run python cli/keyword_search_cli.py bm25idf "dinosaur"

# Inspect BM25 Term Frequency (TF) for a document
uv run python cli/keyword_search_cli.py bm25tf 1 "police"
```

---

### 3. Semantic & Chunked Search

Search documents using dense vector representations:

```bash
# Whole-document semantic search
uv run python cli/semantic_search_cli.py search "toys coming alive when humans leave" --limit 5

# Chunk-level semantic search (better granularity on long plot descriptions)
uv run python cli/semantic_search_cli.py search_chunked "friendship between bear and man in the woods" --limit 5

# Test semantic sentence chunking on sample text
uv run python cli/semantic_search_cli.py semantic_chunk "A young boy discovers a hidden world. He meets a dragon. They become friends." --max-chunk-size 2 --overlap 1
```

---

### 4. Hybrid Search (Weighted & RRF)

Combine lexical and semantic retrieval with optional query enhancement and reranking:

```bash
# Weighted hybrid search (alpha: 0.0 = pure semantic, 1.0 = pure BM25)
uv run python cli/hybrid_search_cli.py weighted-search "superhero with iron suit" --alpha 0.5 --limit 5

# Reciprocal Rank Fusion (RRF) search
uv run python cli/hybrid_search_cli.py rrf-search "time travel paradox" --limit 5

# RRF with LLM query rewrite and Cross-Encoder reranking
uv run python cli/hybrid_search_cli.py rrf-search "that movie with bears and honey" \
  --enhance rewrite \
  --rerank-method cross_encoder \
  --limit 5

# RRF with LLM batch reranking
uv run python cli/hybrid_search_cli.py rrf-search "post apocalyptic desert cars" \
  --enhance expand \
  --rerank-method batch \
  --limit 5
```

---

### 5. Retrieval-Augmented Generation (RAG)

Leverage retrieved movie context to generate answers:

```bash
# Standard RAG answer
uv run python cli/augmented_generation_cli.py rag "What animated movies feature talking bears?"

# Multi-result plot summary for streaming users
uv run python cli/augmented_generation_cli.py summarize "space exploration black hole" --limit 3

# Answer with bracketed citations [1], [2] referencing sources
uv run python cli/augmented_generation_cli.py citations "movies about dream invasion and mind heists" --limit 5

# Conversational Q&A tailored for Webflyx streaming catalog
uv run python cli/augmented_generation_cli.py question "Can you recommend a scary zombie movie set in winter?" --limit 5
```

---

### 6. Multimodal Image Search

Search the catalog using an image query:

```bash
# Search using an image (e.g. data/paddington.jpeg)
uv run python cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Synthesize an image and a text prompt to produce an improved movie search query
uv run python cli/describe_image_cli.py --image data/paddington.jpeg --query "similar british comedy"
```

---

### 7. Evaluation

Benchmark search effectiveness against the golden dataset:

```bash
# Run Precision@k, Recall@k, and F1 evaluation
uv run python cli/evaluation_cli.py --limit 5
```

Sample output:
```text
k=5
- Query: cute british bear marmalade
  - Precision@5: 0.2000
  - Recall@5: 1.0000
  - F1 Score: 0.3333
  - Retrieved: Paddington, The Jungle Book, ...
  - Relevant: Paddington
```

---

## Evaluation Dataset

The evaluation suite (`data/golden_dataset.json`) defines query test cases paired with ground-truth relevant titles across genres:
- Talking bears & animated bear adventures
- Superhero films
- Magic & wizarding worlds
- Zombie apocalypses
- Car racing & high-speed action
- Romantic comedy weddings

---

## Documentation

For an in-depth architectural breakdown, data model analysis, and inventory of technical trade-offs, see the [Product Understanding Document](docs/product_understanding_document.md).
