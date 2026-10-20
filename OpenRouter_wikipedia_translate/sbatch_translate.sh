#!/bin/bash
#SBATCH --job-name=als_wiki_translate
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

# SLURM executes a spool copy; submit from OpenRouter_wikipedia_translate/.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" run_openrouter_wikipedia_translate.py "$@"
