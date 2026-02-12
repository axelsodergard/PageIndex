"""
Local retrieval engine for PageIndex.

4-stage pipeline: Load → Search → Extract → Answer

Usage:
    from pageindex.retrieval import retrieve
    result = retrieve(query, pdf_path, tree_path)
"""

import json
import logging

from .utils import (
    ChatGPT_API,
    ConfigLoader,
    build_node_index,
    count_tokens,
    extract_json,
    get_page_tokens,
    get_subtree,
    get_text_of_pdf_pages,
    get_top_level_nodes,
    structure_to_list,
)
from .prompts_retrieval import (
    build_answer_prompt,
    build_branch_select_prompt,
    build_leaf_refine_prompt,
    build_search_prompt,
)

logger = logging.getLogger(__name__)

_config_loader = ConfigLoader()
_defaults = _config_loader.load()


# ---------------------------------------------------------------------------
# Stage 0 — Load document
# ---------------------------------------------------------------------------

def load_document(pdf_path, tree_path):
    """Load tree JSON and PDF text. Returns a doc dict for the pipeline.

    Returns:
        dict with keys: tree, pdf_pages, node_index, doc_name, total_pages
    """
    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    pdf_pages = get_page_tokens(pdf_path)
    node_index = build_node_index(tree["structure"], pdf_pages)

    return {
        "tree": tree,
        "pdf_pages": pdf_pages,
        "node_index": node_index,
        "doc_name": tree.get("doc_name", "Unknown"),
        "doc_description": tree.get("doc_description", ""),
        "total_pages": len(pdf_pages),
    }


# ---------------------------------------------------------------------------
# Stage 1 — Search tree
# ---------------------------------------------------------------------------

def _build_tree_for_prompt(structure):
    """Strip text fields from structure for prompt inclusion."""
    if isinstance(structure, list):
        return [_build_tree_for_prompt(item) for item in structure]
    elif isinstance(structure, dict):
        result = {}
        for k, v in structure.items():
            if k == "text":
                continue
            if k == "nodes":
                result[k] = _build_tree_for_prompt(v)
            else:
                result[k] = v
        return result
    return structure


def _parse_search_response(raw_response, stage_name):
    """Parse LLM search response and validate structure.

    Args:
        raw_response: Raw string from ChatGPT_API
        stage_name: Human-readable name for error messages

    Returns:
        List of scored node dicts [{node_id, relevance, reason}, ...]

    Raises:
        RuntimeError: If the LLM call returned "Error" or JSON is invalid
    """
    if raw_response == "Error":
        raise RuntimeError(
            f"LLM call failed during {stage_name} — all retries exhausted"
        )

    parsed = extract_json(raw_response)
    if not parsed or "relevant_nodes" not in parsed:
        logger.warning(
            "Search response missing 'relevant_nodes'. Raw: %s",
            raw_response[:500],
        )
        raise RuntimeError(
            f"Invalid search response during {stage_name}: "
            f"missing 'relevant_nodes' key. Raw response logged."
        )

    return parsed["relevant_nodes"]


def _search_single_pass(query, doc, model):
    """Single-pass search for small trees (≤ threshold nodes)."""
    prompt_tree = _build_tree_for_prompt(doc["tree"]["structure"])
    prompt = build_search_prompt(query, doc, prompt_tree)

    raw = ChatGPT_API(model, prompt, temperature=None)
    return _parse_search_response(raw, "single-pass search")


def _search_hierarchical(query, doc, model):
    """Two-phase hierarchical search for large trees.

    Phase 1: Branch selection on top-level nodes.
    Phase 2: Leaf refinement within selected branches.
    """
    structure = doc["tree"]["structure"]

    # Phase 1 — branch select
    top_level = get_top_level_nodes(structure, max_depth=1)
    prompt1 = build_branch_select_prompt(query, doc, top_level)
    raw1 = ChatGPT_API(model, prompt1, temperature=None)
    branch_nodes = _parse_search_response(raw1, "hierarchical branch-select")

    if not branch_nodes:
        return []

    # Collect subtrees for selected branches
    selected_ids = {n["node_id"] for n in branch_nodes}
    subtrees = []
    for node_id in selected_ids:
        subtree = get_subtree(structure, node_id)
        if subtree:
            subtrees.append(subtree)

    if not subtrees:
        # Fall back to branch-level results if subtrees can't be found
        return branch_nodes

    # Phase 2 — leaf refine
    prompt2 = build_leaf_refine_prompt(query, doc, subtrees)
    raw2 = ChatGPT_API(model, prompt2, temperature=None)
    return _parse_search_response(raw2, "hierarchical leaf-refine")


def search_tree(query, doc, model=None, token_budget=None, max_nodes=None):
    """Search the document tree to find relevant sections.

    Adaptive strategy:
      - ≤ threshold nodes → single-pass (one LLM call)
      - > threshold nodes → hierarchical (two LLM calls)

    Returns:
        List of scored nodes sorted by relevance (descending), each with:
        {node_id, title, start_index, end_index, relevance, reason, token_count}
    """
    model = model or _defaults.model
    max_nodes = max_nodes or _defaults.retrieval_max_nodes
    threshold = _defaults.retrieval_tree_threshold

    node_count = len(doc["node_index"])

    if node_count <= threshold:
        strategy = "single_pass"
        raw_nodes = _search_single_pass(query, doc, model)
    else:
        strategy = "hierarchical"
        raw_nodes = _search_hierarchical(query, doc, model)

    # Enrich with data from node_index
    scored = []
    for rn in raw_nodes:
        nid = rn.get("node_id", "")
        if nid not in doc["node_index"]:
            logger.warning("Search returned unknown node_id: %s — skipping", nid)
            continue
        info = doc["node_index"][nid]
        scored.append({
            "node_id": nid,
            "title": info["title"],
            "start_index": info["start_index"],
            "end_index": info["end_index"],
            "relevance": float(rn.get("relevance", 0.5)),
            "reason": rn.get("reason", ""),
            "token_count": info["token_count"],
        })

    # Sort by relevance descending, limit to max_nodes
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    scored = scored[:max_nodes]

    return scored, strategy


# ---------------------------------------------------------------------------
# Stage 2 — Select and extract text
# ---------------------------------------------------------------------------

def _extract_node_text(pdf_pages, start_index, end_index, page_set=None):
    """Extract text for a node's pages with --- Page N --- markers.

    Args:
        pdf_pages: List of (text, token_count) tuples (0-indexed)
        start_index: 1-based start page
        end_index: 1-based end page
        page_set: Optional set of already-included pages for deduplication.
                  Pages in this set will be skipped. New pages are added.

    Returns:
        (text, tokens_added, pages_added) tuple
    """
    if page_set is None:
        page_set = set()

    text_parts = []
    tokens = 0
    pages_added = []
    for page_num in range(start_index, end_index + 1):
        if page_num in page_set:
            continue
        idx = page_num - 1  # 1-based → 0-indexed
        if idx < 0 or idx >= len(pdf_pages):
            continue
        page_text, page_tokens = pdf_pages[idx]
        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        tokens += page_tokens
        pages_added.append(page_num)
        page_set.add(page_num)

    return "\n\n".join(text_parts), tokens, pages_added


def _truncate_node(pdf_pages, start_index, end_index, token_limit, page_set):
    """Extract pages front-to-back until token budget is hit.

    Known limitation: front-to-back truncation biases toward section preambles.
    For regulatory documents, detailed requirements often appear later in a section.

    Returns:
        (text, tokens_used, pages_added, was_truncated) tuple
    """
    text_parts = []
    tokens = 0
    pages_added = []
    was_truncated = False

    for page_num in range(start_index, end_index + 1):
        if page_num in page_set:
            continue
        idx = page_num - 1
        if idx < 0 or idx >= len(pdf_pages):
            continue
        page_text, page_tokens = pdf_pages[idx]
        if tokens + page_tokens > token_limit:
            was_truncated = True
            break
        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        tokens += page_tokens
        pages_added.append(page_num)
        page_set.add(page_num)

    return "\n\n".join(text_parts), tokens, pages_added, was_truncated


def select_and_extract(scored_nodes, doc, token_budget=None):
    """Greedy knapsack: select nodes by relevance, extract text within budget.

    - Deduplicates overlapping pages (parent/child ranges)
    - If the top node exceeds budget, truncates it
    - Re-sorts selected nodes by start_index for coherent reading order

    Returns:
        dict with: context_text, selected_nodes, total_context_tokens,
                   skipped_nodes, was_truncated
    """
    token_budget = token_budget or _defaults.retrieval_token_budget
    pdf_pages = doc["pdf_pages"]

    included_pages = set()
    selected = []
    skipped = []
    total_tokens = 0
    was_truncated = False

    for i, node in enumerate(scored_nodes):
        # Estimate actual tokens after dedup
        dedup_tokens = 0
        for p in range(node["start_index"], node["end_index"] + 1):
            if p not in included_pages:
                idx = p - 1
                if 0 <= idx < len(pdf_pages):
                    dedup_tokens += pdf_pages[idx][1]

        if dedup_tokens == 0:
            # All pages already included
            continue

        remaining = token_budget - total_tokens

        if dedup_tokens <= remaining:
            # Fits within budget — extract all pages
            text, tokens, pages = _extract_node_text(
                pdf_pages, node["start_index"], node["end_index"], included_pages
            )
            if tokens > 0:
                selected.append({**node, "_text": text, "_tokens": tokens, "_pages": pages})
                total_tokens += tokens
        elif i == 0 and remaining > 0:
            # First (highest relevance) node exceeds budget — truncate it
            text, tokens, pages, trunc = _truncate_node(
                pdf_pages, node["start_index"], node["end_index"],
                remaining, included_pages
            )
            if tokens > 0:
                selected.append({**node, "_text": text, "_tokens": tokens, "_pages": pages})
                total_tokens += tokens
                was_truncated = was_truncated or trunc
        else:
            skipped.append(node)

    # Re-sort selected by start_index for coherent reading order
    selected.sort(key=lambda x: x["start_index"])

    # Assemble context text
    context_parts = []
    for s in selected:
        context_parts.append(s["_text"])
    context_text = "\n\n".join(context_parts)

    # Clean internal fields from selected for output
    clean_selected = []
    for s in selected:
        clean_selected.append({
            k: v for k, v in s.items() if not k.startswith("_")
        })

    return {
        "context_text": context_text,
        "selected_nodes": clean_selected,
        "total_context_tokens": total_tokens,
        "skipped_nodes": skipped,
        "was_truncated": was_truncated,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Generate answer
# ---------------------------------------------------------------------------

def generate_answer(query, context, doc, model=None):
    """Send assembled context + query to LLM for answer generation.

    Falls back to raw text as answer if JSON parsing fails.

    Returns:
        dict with: answer, pages_cited, confidence
    """
    model = model or _defaults.model
    prompt = build_answer_prompt(
        query, doc, context["context_text"], context["was_truncated"]
    )

    raw = ChatGPT_API(model, prompt, temperature=None)

    if raw == "Error":
        raise RuntimeError(
            "LLM call failed during answer generation — all retries exhausted"
        )

    parsed = extract_json(raw)

    if parsed and "answer" in parsed:
        return {
            "answer": parsed["answer"],
            "pages_cited": parsed.get("pages_cited", []),
            "confidence": parsed.get("confidence", "unknown"),
        }

    # Graceful fallback: use raw text as answer
    logger.warning("Answer response was not valid JSON — using raw text as answer")
    return {
        "answer": raw,
        "pages_cited": [],
        "confidence": "unknown",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def retrieve(query, pdf_path, tree_path, model=None, token_budget=None,
             max_nodes=None, verbose=False):
    """Main retrieval function. Runs the full Load → Search → Extract → Answer pipeline.

    Args:
        query: User question
        pdf_path: Path to the PDF file
        tree_path: Path to the tree JSON file
        model: LLM model name (default from config.yaml)
        token_budget: Max tokens for retrieved context (default 150000)
        max_nodes: Max nodes to consider from search (default 15)
        verbose: Print human-readable summary to stdout

    Returns:
        Complete result dict with answer, sources, and search trace
    """
    model = model or _defaults.model
    token_budget = token_budget or _defaults.retrieval_token_budget
    max_nodes = max_nodes or _defaults.retrieval_max_nodes

    # Stage 0 — Load
    if verbose:
        print(f"Loading document: {pdf_path}")
    doc = load_document(pdf_path, tree_path)
    if verbose:
        print(f"  Loaded {doc['total_pages']} pages, {len(doc['node_index'])} nodes")

    # Stage 1 — Search
    if verbose:
        print(f"Searching tree for: {query}")
    scored_nodes, strategy = search_tree(query, doc, model, token_budget, max_nodes)
    if verbose:
        print(f"  Strategy: {strategy}, found {len(scored_nodes)} relevant nodes")

    if not scored_nodes:
        return {
            "query": query,
            "answer": "No relevant sections found in the document for this query.",
            "confidence": "not_found",
            "pages_cited": [],
            "sources": [],
            "search_trace": {
                "strategy": strategy,
                "nodes_in_tree": len(doc["node_index"]),
                "nodes_scored": 0,
                "nodes_selected": 0,
                "nodes_skipped": 0,
                "context_tokens": 0,
            },
        }

    # Stage 2 — Select and extract
    if verbose:
        print(f"Extracting text (budget: {token_budget} tokens)")
    context = select_and_extract(scored_nodes, doc, token_budget)
    if verbose:
        print(f"  Selected {len(context['selected_nodes'])} nodes, "
              f"{context['total_context_tokens']} tokens")
        if context["skipped_nodes"]:
            print(f"  Skipped {len(context['skipped_nodes'])} nodes (over budget)")

    # Stage 3 — Answer
    if verbose:
        print("Generating answer...")
    answer_result = generate_answer(query, context, doc, model)

    # Assemble result
    sources = []
    for node in context["selected_nodes"]:
        sources.append({
            "node_id": node["node_id"],
            "title": node["title"],
            "pages": f"{node['start_index']}-{node['end_index']}",
            "relevance": node["relevance"],
            "reason": node["reason"],
        })

    result = {
        "query": query,
        "answer": answer_result["answer"],
        "confidence": answer_result["confidence"],
        "pages_cited": answer_result["pages_cited"],
        "sources": sources,
        "search_trace": {
            "strategy": strategy,
            "nodes_in_tree": len(doc["node_index"]),
            "nodes_scored": len(scored_nodes),
            "nodes_selected": len(context["selected_nodes"]),
            "nodes_skipped": len(context["skipped_nodes"]),
            "context_tokens": context["total_context_tokens"],
        },
    }

    # Print human-readable output
    if verbose:
        _print_result(result)

    return result


def _print_result(result):
    """Print a human-readable summary of the retrieval result."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']}")
    if result["pages_cited"]:
        print(f"Pages cited: {', '.join(str(p) for p in result['pages_cited'])}")

    print("\n" + "-" * 60)
    print("SOURCES")
    print("-" * 60)
    print(f"{'Node':<8} {'Pages':<10} {'Rel':>5}  Title / Reason")
    print("-" * 60)
    for src in result["sources"]:
        print(f"{src['node_id']:<8} {src['pages']:<10} {src['relevance']:>5.2f}  {src['title']}")
        if src["reason"]:
            print(f"{'':>25} {src['reason']}")

    trace = result["search_trace"]
    print("\n" + "-" * 60)
    print("SEARCH TRACE")
    print("-" * 60)
    print(f"  Strategy:       {trace['strategy']}")
    print(f"  Nodes in tree:  {trace['nodes_in_tree']}")
    print(f"  Nodes scored:   {trace['nodes_scored']}")
    print(f"  Nodes selected: {trace['nodes_selected']}")
    print(f"  Nodes skipped:  {trace['nodes_skipped']}")
    print(f"  Context tokens: {trace['context_tokens']}")
    print("=" * 60)
