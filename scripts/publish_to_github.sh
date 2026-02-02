#!/usr/bin/env bash
set -euo pipefail
set -a; source .env; set +a

: "${GITHUB_REPO_SSH:?Need GITHUB_REPO_SSH}"
: "${PUBLIC_STRIP_PATH:=results/plots/persrec_tc5_2025-08-21/}"

workdir="${1:-/tmp/repo-public-mirror}"

rm -rf "$workdir"
git clone --mirror "${CI_REPOSITORY_URL}" "$workdir"
cd "$workdir"

git remote add github "$GITHUB_REPO_SSH"

# Remove secret/ from ALL commits
git filter-repo --path "$PUBLIC_STRIP_PATH" --invert-paths --force

# Push all refs
git push --force --mirror github
