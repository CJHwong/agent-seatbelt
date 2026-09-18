#!/usr/bin/env bash
# The hook suite moved to hooks-pii/, because "opf" names one detection backend
# and this is the whole suite.
#
# This file exists so that every install command already written down keeps
# working. GitHub's raw URL does not redirect, so a stub at the old path is the
# only way to keep them alive. It fetches the current installer and passes the
# arguments through, so --prompt-only, --no-codex and --no-pilot behave the same
# from either URL.
#
# Nothing else lives here. Do not add to this directory.

set -euo pipefail

REPO_BASE="${HOOKS_PII_BASE_URL:-${HOOKS_OPF_BASE_URL:-https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-pii}}"

echo "hooks-opf has moved to hooks-pii. Fetching the current installer." >&2
curl -fsSL "$REPO_BASE/install.sh" | bash -s -- "$@"
