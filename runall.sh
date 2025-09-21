#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_all_notebooks.sh [root_dir]
ROOT_FOLDER="${1:-.}"

# Update the virtual environment:
echo "Synchronizing virtual environment..."
uv sync

echo "Removing checkopoint files..."
rm -vrf .ipynb_checkpoints

# Restart kernel, execute all cells, and save in place for every .ipynb
echo "Running all the notebooks..."
find "${ROOT_FOLDER}" -type f -name '*.ipynb' \
  | sort  \
  | while IFS= read -r nb; do
      echo "Executing: $nb"
      uv run jupyter nbconvert --to notebook --execute "$nb" --inplace
    done
