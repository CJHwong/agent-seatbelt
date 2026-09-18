#!/usr/bin/env bash
# coverage.sh — line coverage for the shipped bash, in pure bash.
#
# Runs the offline suite under `set -x` with a PS4 that stamps every executed
# command with its file and line, then reports covered/executable lines per file
# and enforces a floor so coverage can only ratchet up.
#
#   ./coverage.sh                        report
#   PII_COV_FLOOR=100 ./coverage.sh      report and fail if any file is under 100%
#   PII_COV_DETAIL=1 ./coverage.sh       list the uncovered lines
#   PII_COV_AUDIT=1 ./coverage.sh        show where the static estimate drifted
#
# The mechanism is ported from slacker.sh's .dev/tests/coverage.sh, as is
# execlines.awk next to this file. No new dependency: PS4 with
# ${BASH_SOURCE}/${LINENO} behaves the same on bash 3.2 and 5.x, and kcov cannot
# instrument bash 3.2 at all.
#
# One difference from the original. There, the traced children are bash functions
# called from the runner. Here the children are Python subprocesses, so the trace
# cannot live on an inherited fd set up by the runner. cov_env.sh is handed to
# each child through BASH_ENV instead, and the child opens the trace itself.
#
# The floor applies to each file, not to the total. A total lets one file improve
# while another regresses and still passes, which is the opposite of a ratchet.
#
# The suite table prints on every run, pass or fail. Its summary line carries each
# suite's skip count, and a green run that quietly skipped a suite must not read as
# one that checked it. The canary suite is the case that made this necessary: it
# skips wherever the CLI it drives is absent, which is every CI runner by default.
#
# What this does NOT measure:
#   - the Python side (pii-server.py, pii_rules.py, pii_opf.py, pii_redact_torch.py,
#     pii_redact_lite.py).
#     Use `uv run --with coverage python -m coverage run --branch` for those.
#   - branch coverage. A two-arm check counts as one covered line. Use the Python
#     command above with --branch where it matters; a bash guard whose false arm no
#     test takes looks identical to a covered line here.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# The bash this project ships. Both are driven by the offline tests, so one trace
# run covers them.
TARGETS=("hooks-pii/pii-check.sh" "hooks-pii/install.sh")
# The suites that need no model, which is also the set CI runs on a pull request.
# Deliberately not test_*.py: test_pii_opf and test_pii_redact_torch need torch and
# onnxruntime, so they would either crawl or fail on a runner without them.
TEST_PATTERNS=(
    "test_hook_*.py"
    "test_install.py"
    "test_pii_rules.py"
    "test_pii_server.py"
)

TRACE=$(mktemp "${TMPDIR:-/tmp}/pii_cov.XXXXXX")
HITS=$(mktemp "${TMPDIR:-/tmp}/pii_hits.XXXXXX")
SUITE_LOG=$(mktemp "${TMPDIR:-/tmp}/pii_suite.XXXXXX")
HITS_EXEC="$HITS.exec"
HITS_GOT="$HITS.got"
trap 'rm -f "$TRACE" "$HITS" "$HITS_EXEC" "$HITS_GOT" "$SUITE_LOG"' EXIT INT TERM

export PII_COV_TRACE="$TRACE"
export BASH_ENV="$DIR/cov_env.sh"

# The bash tests are stdlib-only, so they run on the system python3 rather than
# through `uv run`, which would resolve the model dependencies they never touch.
#
# Each suite's output is captured rather than discarded. The trace has to stay
# quiet, but hiding a failure too would leave a CI run saying only that something
# broke, which is worse than saying nothing at all.
suite_status=0
suite_rows=""
for pattern in "${TEST_PATTERNS[@]}"; do
    one_log=$(mktemp "${TMPDIR:-/tmp}/pii_one.XXXXXX")
    if ! python3 -m unittest discover -s "$DIR" -p "$pattern" >"$one_log" 2>&1; then
        suite_status=1
        echo "pii-check coverage: the suite matching '$pattern' failed" >&2
    fi
    cat "$one_log" >>"$SUITE_LOG"
    # One row per suite, always printed. The summary line carries the skip count, and
    # a suite that skipped itself must say so on a green run too. Otherwise "passed"
    # reads as "checked", and the suite that matters most here is the one that skips
    # wherever the CLI it drives is absent.
    suite_rows="$suite_rows$(awk -v p="$pattern" '
        /^Ran [0-9]+ tests?/ { ran = $2 }
        /^(OK|FAILED)/      { result = $0 }
        END { printf "%-24s %6s   %s", p, (ran ? ran : "-"), (result ? result : "no result line") }
    ' "$one_log")
"
    rm -f "$one_log"
done

printf '%-24s %6s   %s\n' SUITE TESTS RESULT
printf '%s' "$suite_rows"
echo

# Merge every record and normalize the absolute path down to the repo-relative
# name, so a record for either target lands on the key the loop below looks up.
grep -ao 'COV:[^:]*:[0-9]*:' "$TRACE" 2>/dev/null \
  | sed -e 's/^COV://' -e 's/:$//' \
        -e 's|^.*/hooks-pii/||' \
  | grep -E '^(pii-check\.sh|install\.sh):' \
  | sed -e 's|^pii-check\.sh:|hooks-pii/pii-check.sh:|' \
        -e 's|^install\.sh:|hooks-pii/install.sh:|' \
  | sort -u > "$HITS_GOT"

printf '%-28s %8s %8s %7s\n' FILE COVERED EXEC PCT
printf '%-28s %8s %8s %7s\n' '----' '-------' '----' '---'

total_hit=0
total_exec=0
audit=""
detail=""
floor_failures=""
floor="${PII_COV_FLOOR:-0}"

for target in "${TARGETS[@]}"; do
    # Denominator = the static estimate UNION the lines bash actually traced.
    #
    # Static analysis alone cannot be exact: bash reports a multi-line command at
    # the line where it ends, so a `$( … )` spanning four lines is traced against
    # its closing line. Anything observed is executable by proof, so unioning it in
    # can only make the denominator more correct, and it guarantees covered never
    # exceeds total.
    awk -f "$DIR/execlines.awk" -v mode=list "$ROOT/$target" | sort -u > "$HITS_EXEC"
    grep "^$target:" "$HITS_GOT" | sed "s|^$target:||" | sort -u > "$HITS_GOT.one"
    leak=$(comm -13 "$HITS_EXEC" "$HITS_GOT.one" | tr '\n' ' ')
    [ -n "$leak" ] && audit="$audit  $target: only observed, not predicted: $leak
"

    exec_n=$(sort -u "$HITS_EXEC" "$HITS_GOT.one" | grep -c .)
    hit_n=$(grep -c . "$HITS_GOT.one")
    total_hit=$((total_hit + hit_n))
    total_exec=$((total_exec + exec_n))
    # awk's own printf, not the shell's: the row ends in "%", which the shell would
    # try to read as a format directive.
    awk -v f="$target" -v h="$hit_n" -v d="$exec_n" \
        'BEGIN{printf "%-28s %8d %8d %6.1f%%\n", f, h, d, (d ? 100*h/d : 100)}'

    if awk -v p="$(awk -v h="$hit_n" -v d="$exec_n" 'BEGIN{printf "%.2f", (d ? 100*h/d : 100)}')" \
           -v f="$floor" 'BEGIN{exit !(p + 0.005 < f)}'; then
        floor_failures="$floor_failures  $target
"
    fi

    if [ -n "${PII_COV_DETAIL:-}" ]; then
        miss=$(comm -23 "$HITS_EXEC" "$HITS_GOT.one")
        if [ -n "$miss" ]; then
            detail="$detail
uncovered lines in $target:
$(while read -r line_number; do
    [ -n "$line_number" ] || continue
    printf '  %s:%s: %s\n' "$target" "$line_number" "$(sed -n "${line_number}p" "$ROOT/$target")"
  done <<< "$miss")
"
        fi
    fi
done
rm -f "$HITS_GOT.one"

pct=$(awk -v h="$total_hit" -v d="$total_exec" 'BEGIN{printf "%.1f", (d ? 100*h/d : 100)}')
printf '%-28s %8s %8s %7s\n' '----' '-------' '----' '---'
awk -v h="$total_hit" -v d="$total_exec" \
    'BEGIN{printf "%-28s %8d %8d %6.1f%%\n", "TOTAL", h, d, (d ? 100*h/d : 100)}'

if [ -n "${PII_COV_AUDIT:-}" ] && [ -n "$audit" ]; then
    echo
    echo "execlines.awk gap (observed but not predicted; folded into the total):"
    printf '%s' "$audit"
fi

if [ -n "${PII_COV_DETAIL:-}" ] && [ -n "$detail" ]; then
    printf '%s\n' "$detail"
fi

if [ "$suite_status" -ne 0 ]; then
    echo
    echo "== a test suite failed; the numbers above are not trustworthy =="
    echo "== the failing suite's output follows =="
    cat "$SUITE_LOG"
    exit 1
fi

if [ -n "$floor_failures" ]; then
    echo
    echo "== coverage below the floor of ${floor}% in:"
    printf '%s' "$floor_failures"
    exit 1
fi
echo
echo "== coverage ${pct}% (floor ${floor}% per file) =="
