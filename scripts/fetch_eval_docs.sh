#!/usr/bin/env bash
# Downloads the three arXiv papers the eval uses into docs/ (they are not committed to the repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs
fetch() { [ -s "docs/$2.pdf" ] && echo "have $2.pdf" || { echo "downloading $2.pdf"; curl -sL -o "docs/$2.pdf" "https://arxiv.org/pdf/$1"; }; }
fetch 1706.03762 attention   # Attention Is All You Need
fetch 2005.11401 rag         # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
fetch 1810.04805 bert        # BERT
