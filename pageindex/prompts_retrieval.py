"""Prompt templates and builder functions for the retrieval pipeline."""

import json


# --- Search prompts ---

TREE_SEARCH_SINGLE_PASS = """You are a document analysis expert. You are given the structure of a document and a user query. Your task is to identify which sections of the document are most likely to contain information relevant to answering the query.

## Document: {doc_name}
{doc_description}

## Document Structure
{tree_structure}

## User Query
{query}

## Instructions
Analyze the document structure and identify sections relevant to the query. Consider:
- Section titles and their summaries
- Page ranges (sections with more pages may contain more detail)
- Parent-child relationships (a relevant parent may have relevant children)

Return your analysis as JSON:
```json
{{
    "thinking": "Brief explanation of your reasoning about which sections are relevant and why",
    "relevant_nodes": [
        {{
            "node_id": "0001",
            "relevance": 0.95,
            "reason": "Cite specific words from the section title or summary that connect to the query. Do not speculate — only reference what you can see in the structure."
        }}
    ]
}}
```

Rules:
- `relevance` is a float from 0.0 to 1.0 indicating how likely the section contains relevant information
- Only include nodes with relevance >= 0.3
- Order nodes by relevance (highest first)
- Prefer leaf nodes over parent nodes when the leaf clearly covers the topic
- If a parent node is relevant but you also select its children, give the children higher relevance scores since they provide more focused content
"""

TREE_SEARCH_BRANCH_SELECT = """You are a document analysis expert. You are given the top-level structure of a large document and a user query. Your task is to identify which branches (top-level sections) are most likely to contain information relevant to the query.

## Document: {doc_name}
{doc_description}

## Top-Level Structure
{tree_structure}

## User Query
{query}

## Instructions
Select the branches most likely to contain relevant information. This is a coarse selection — we will examine the selected branches in more detail next.

Return your analysis as JSON:
```json
{{
    "thinking": "Brief reasoning about which branches to explore",
    "relevant_nodes": [
        {{
            "node_id": "0001",
            "relevance": 0.95,
            "reason": "Cite specific words from the section title or summary that connect to the query. Do not speculate — only reference what you can see in the structure."
        }}
    ]
}}
```

Rules:
- `relevance` is a float from 0.0 to 1.0
- Only include nodes with relevance >= 0.2 (lower threshold since this is coarse selection)
- Order by relevance (highest first)
- When in doubt, include a branch — it's better to examine too many than too few
"""

TREE_SEARCH_LEAF_REFINE = """You are a document analysis expert. You are refining a search within selected branches of a document. You are given detailed subtrees and a user query. Identify the specific sections most relevant to the query.

## Document: {doc_name}
{doc_description}

## Selected Branches (detailed)
{tree_structure}

## User Query
{query}

## Instructions
Identify the specific sections (preferably leaf nodes) most relevant to the query.

Return your analysis as JSON:
```json
{{
    "thinking": "Brief reasoning about which specific sections are most relevant",
    "relevant_nodes": [
        {{
            "node_id": "0015",
            "relevance": 0.95,
            "reason": "Cite specific words from the section title or summary that connect to the query. Do not speculate — only reference what you can see in the structure."
        }}
    ]
}}
```

Rules:
- `relevance` is a float from 0.0 to 1.0
- Only include nodes with relevance >= 0.3
- Order by relevance (highest first)
- Prefer leaf nodes over parent nodes when the leaf clearly covers the topic
- If a parent and its child are both relevant, prefer the child unless the parent contains unique introductory content
"""

ANSWER_GENERATION = """You are an expert analyst. Answer the user's question based solely on the provided document content. Cite specific pages and include short verbatim quotes to support your answer.

## Document: {doc_name}

## Retrieved Content
{context}

## User Query
{query}

## Instructions
- Open with 1-2 sentences stating who or what is in scope and under which regulation, guideline, or framework, to orient the reader before listing detailed requirements.
- For each requirement or claim, include a short exact quote (25 words or fewer) from the source text, formatted as: > "exact words from the document" [Page N]
- Prefix each numbered requirement with an obligation tag based on the language in the source text:
  - [MUST] for mandatory obligations (source uses "shall", "must", "are required to")
  - [SHOULD] for strong recommendations (source uses "should", "are expected to")
  - [MAY] for optional or permissive items (source uses "may", "can", "is allowed to")
  - When the obligation level is unclear, default to [SHOULD].
- Use only the provided content to answer. If the content does not contain enough information to fully answer the question, say so explicitly.
- Be thorough but concise.

{truncation_notice}

Return your response as JSON:
```json
{{
    "answer": "Your detailed answer with scope preface, obligation tags, verbatim quotes, and [Page N] citations...",
    "pages_cited": [31, 33, 35],
    "confidence": "high"
}}
```

Confidence levels:
- "high": Every claim is backed by a verbatim quote from the retrieved content
- "medium": The content answers the question but some claims rely on interpretation rather than direct quotes
- "low": The content is tangentially related but does not directly address the question
- "not_found": The retrieved content does not contain relevant information to answer the question
"""

TRUNCATION_NOTICE = "Note: Some retrieved content was truncated due to length constraints. The answer should be based on the available content."


# --- Builder functions ---

def _format_node_for_prompt(node, depth=0):
    """Format a single node for display in prompts."""
    indent = "  " * depth
    parts = [f"{indent}- [{node.get('node_id', '?')}] {node.get('title', 'Untitled')}"]
    parts.append(f" (pages {node.get('start_index', '?')}-{node.get('end_index', '?')})")
    if node.get('summary'):
        # Truncate long summaries for prompt efficiency
        summary = node['summary']
        if len(summary) > 300:
            summary = summary[:300] + "..."
        parts.append(f"\n{indent}  Summary: {summary}")
    return "".join(parts)


def _format_tree_for_prompt(structure, depth=0):
    """Recursively format tree structure for prompt display."""
    lines = []
    if isinstance(structure, list):
        for node in structure:
            lines.extend(_format_tree_for_prompt(node, depth))
    elif isinstance(structure, dict):
        lines.append(_format_node_for_prompt(structure, depth))
        if structure.get('nodes'):
            lines.extend(_format_tree_for_prompt(structure['nodes'], depth + 1))
    return lines


def build_search_prompt(query, doc, tree_structure):
    """Build a single-pass search prompt."""
    tree_text = "\n".join(_format_tree_for_prompt(tree_structure))
    doc_description = ""
    if doc.get("doc_description"):
        doc_description = f"Description: {doc['doc_description']}"
    return TREE_SEARCH_SINGLE_PASS.format(
        doc_name=doc.get("doc_name", "Unknown"),
        doc_description=doc_description,
        tree_structure=tree_text,
        query=query,
    )


def build_branch_select_prompt(query, doc, top_level_structure):
    """Build a branch selection prompt for hierarchical search phase 1."""
    tree_text = "\n".join(_format_tree_for_prompt(top_level_structure))
    doc_description = ""
    if doc.get("doc_description"):
        doc_description = f"Description: {doc['doc_description']}"
    return TREE_SEARCH_BRANCH_SELECT.format(
        doc_name=doc.get("doc_name", "Unknown"),
        doc_description=doc_description,
        tree_structure=tree_text,
        query=query,
    )


def build_leaf_refine_prompt(query, doc, subtrees):
    """Build a leaf refinement prompt for hierarchical search phase 2."""
    tree_text = "\n".join(_format_tree_for_prompt(subtrees))
    doc_description = ""
    if doc.get("doc_description"):
        doc_description = f"Description: {doc['doc_description']}"
    return TREE_SEARCH_LEAF_REFINE.format(
        doc_name=doc.get("doc_name", "Unknown"),
        doc_description=doc_description,
        tree_structure=tree_text,
        query=query,
    )


def build_answer_prompt(query, doc, context_text, was_truncated=False):
    """Build an answer generation prompt."""
    truncation_notice = TRUNCATION_NOTICE if was_truncated else ""
    return ANSWER_GENERATION.format(
        doc_name=doc.get("doc_name", "Unknown"),
        context=context_text,
        query=query,
        truncation_notice=truncation_notice,
    )
