# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PageIndex is a vectorless, reasoning-based RAG system by Vectify AI. It transforms PDF and Markdown documents into hierarchical tree structures (like a table of contents) using LLM reasoning — no vector DB or chunking. The tree index enables agentic, context-aware retrieval over long documents.

## Commands

```bash
# Install dependencies
pip3 install --upgrade -r requirements.txt

# Process a PDF
python3 run_pageindex.py --pdf_path /path/to/document.pdf

# Process a Markdown file
python3 run_pageindex.py --md_path /path/to/document.md
```

Key CLI options: `--model`, `--toc-check-pages`, `--max-pages-per-node`, `--max-tokens-per-node`, `--if-add-node-id`, `--if-add-node-summary`, `--if-add-doc-description`, `--if-add-node-text`, `--if-thinning` (markdown only).

There is no formal test suite or linter configured. Testing is done manually via Jupyter notebooks in `cookbook/`.

## Environment

Requires a `.env` file in the project root with:
```
CHATGPT_API_KEY=your_openai_key_here
```

Default config is in `pageindex/config.yaml`. CLI args and programmatic `config()` calls override these defaults.

## Architecture

### Core Modules (`pageindex/`)

- **`page_index.py`** — Main PDF processing pipeline. Entry point: `page_index_main(pdf_path, opt)`. Handles TOC detection, extraction, transformation to JSON tree, page boundary mapping, and validation/correction of section boundaries.
- **`page_index_md.py`** — Markdown processing pipeline. Entry point: `md_to_tree()` (async). Parses markdown headers into a hierarchy, with optional tree thinning to merge small sections.
- **`utils.py`** — Shared utilities: OpenAI API wrappers (`ChatGPT_API`, `ChatGPT_API_async`) with retry logic, token counting via tiktoken, JSON extraction from LLM responses, node ID assignment, summary generation, and `ConfigLoader` for YAML-based config.

### Processing Pipeline (PDF)

1. Extract page text using PyMuPDF/PyPDF2
2. Count tokens per page (tiktoken)
3. Detect and extract table of contents (or generate structure from headings if no TOC)
4. Transform TOC into JSON tree with page boundaries
5. Validate and fix section boundaries via LLM
6. Post-process: assign node IDs, generate summaries, add document description

### Output Format

JSON tree with `doc_name`, optional `doc_description`, and nested `structure` array. Each node has `title`, `node_id`, `start_index`/`end_index` (page numbers), optional `summary`, and nested `nodes`.

### Key Patterns

- Heavy use of async/await with `asyncio.gather()` for parallel LLM calls (summaries, validation)
- LLM-driven processing: GPT-4o is used for TOC extraction, structure generation, boundary verification, and summary generation
- API calls have retry logic (up to 10 retries) in `ChatGPT_API` / `ChatGPT_API_async`
- JSON logging to `./logs/` directory for debugging processing steps

### Entry Point

`run_pageindex.py` — CLI wrapper that parses args, creates config, calls the appropriate pipeline (PDF or Markdown), and saves output JSON to `./results/`.

## Retrieval Pipeline

The retrieval engine (`pageindex/retrieval.py`) provides a local, production-quality query pipeline over indexed documents. It follows a 4-stage architecture: **Load → Search → Extract → Answer**.

### Running Queries

```bash
python run_retrieval.py \
  --pdf_path "tests/pdfs/doc.pdf" \
  --tree_path "results/doc_structure.json" \
  --query "What are the screening requirements?" \
  --verbose
```

Key CLI options:
- `--model` — LLM model (default from config.yaml)
- `--token-budget` — Max tokens for retrieved context (default: 150000)
- `--max-nodes` — Max nodes to consider from search (default: 15)
- `--output` — Save result as JSON file
- `--verbose` — Print detailed search trace

### Model Context Window Requirement

The retrieval pipeline uses a default token budget of 150K tokens for retrieved content. The deployed model must have a context window of at least 200K tokens to accommodate prompt overhead and answer generation. `gpt-4o` (128K) may be tight for large documents; `gpt-4.1` (1M) or `o3` models are recommended. The budget is configurable via `--token-budget`.

### Core Modules

- **`retrieval.py`** — Pipeline stages: `load_document()`, `search_tree()` (adaptive single-pass or hierarchical), `select_and_extract()` (greedy knapsack with page dedup), `generate_answer()`, and the `retrieve()` orchestrator.
- **`prompts_retrieval.py`** — Prompt templates and builder functions for search and answer generation.
