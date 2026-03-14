#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model, embed_text, search_command
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify Semantic search embedding model loading")

    embed_parser = subparsers.add_parser("embed", help="Embeds given text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query = subparsers.add_parser("embedquery", help="Embeds given text")
    embed_query.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search the given search term")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", default=5, type=int, help="Number of results to return")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed":
            embed_text(args.text)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()