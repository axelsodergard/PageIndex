import argparse
import json
import os

from pageindex.retrieval import retrieve
from pageindex.utils import ConfigLoader

if __name__ == "__main__":
    config_loader = ConfigLoader()
    defaults = config_loader.load()

    parser = argparse.ArgumentParser(
        description="Query a PDF document using its PageIndex tree structure"
    )
    parser.add_argument("--pdf_path", type=str, required=True,
                        help="Path to the PDF file")
    parser.add_argument("--tree_path", type=str, required=True,
                        help="Path to the tree structure JSON file")
    parser.add_argument("--query", type=str, required=True,
                        help="Question to answer from the document")
    parser.add_argument("--model", type=str, default=defaults.model,
                        help="LLM model to use")
    parser.add_argument("--token-budget", type=int,
                        default=defaults.retrieval_token_budget,
                        help="Max tokens for retrieved context")
    parser.add_argument("--max-nodes", type=int,
                        default=defaults.retrieval_max_nodes,
                        help="Max nodes to consider from search")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: print to stdout)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed search trace")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.pdf_path):
        raise ValueError(f"PDF file not found: {args.pdf_path}")
    if not os.path.isfile(args.tree_path):
        raise ValueError(f"Tree file not found: {args.tree_path}")

    result = retrieve(
        query=args.query,
        pdf_path=args.pdf_path,
        tree_path=args.tree_path,
        model=args.model,
        token_budget=args.token_budget,
        max_nodes=args.max_nodes,
        verbose=args.verbose,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {args.output}")
    elif not args.verbose:
        # If not verbose, print a concise summary
        print("\nANSWER:", result["answer"])
        print(f"\nConfidence: {result['confidence']}")
        if result["pages_cited"]:
            print(f"Pages cited: {', '.join(str(p) for p in result['pages_cited'])}")
        print(f"\nSources: {len(result['sources'])} nodes selected")
