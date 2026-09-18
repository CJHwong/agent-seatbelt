# shellcheck shell=bash
# cov_env.sh — BASH_ENV shim, read by every non-interactive bash the tests spawn.
#
# Two independent jobs, both gated on an environment variable so an unset var
# leaves the shell untouched:
#
#   PII_COV_TRACE          append an xtrace record to that file, one line per
#                          executed command, stamped with file and line. This is
#                          how coverage.sh sees inside the hook. Exported PS4 is
#                          NOT portable (bash 5.2 ignores it), so it is set here
#                          where the shell that will use it is the one reading it.
#   PII_COV_HIDE_COMMANDS  colon-separated names for which `command -v` must fail.
#
# The second job exists because pii-check.sh:29 prepends /opt/homebrew/bin to
# PATH before the dependency checks, so `command -v jq` can never fail on a
# machine that has jq. Hiding the name is the only way to reach the four
# missing-dependency branches. The shadow defers anything it is not asked to
# hide, so a traced run still finds the real tools.
#
# bash 4.1 added BASH_XTRACEFD, which keeps the trace out of stderr. Older shells
# fall back to pointing stderr at the trace file, which means the suite's stderr
# assertions fail under coverage on those shells. Run the suite directly for
# those; see coverage.sh.

if [ -n "${PII_COV_TRACE:-}" ]; then
    if [ "${BASH_VERSINFO[0]}" -gt 4 ] ||
       { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 1 ]; }; then
        exec 9>>"$PII_COV_TRACE"
        export BASH_XTRACEFD=9
    else
        exec 2>>"$PII_COV_TRACE"
    fi
    # shellcheck disable=SC2016  # must reach the traced shell unexpanded
    #
    # The default matters. A script read from stdin, which is how `curl | bash`
    # installs itself, has no BASH_SOURCE at all, and under `set -u` the prompt
    # expansion then kills the traced shell with "BASH_SOURCE: unbound variable"
    # before its first line runs. The records such a shell does emit name "stdin",
    # which no target matches, so they fall out of the per-file count.
    PS4='+COV:${BASH_SOURCE[0]:-stdin}:${LINENO}:'
    set -x
fi

if [ -n "${PII_COV_HIDE_COMMANDS:-}" ]; then
    # shellcheck disable=SC2317  # invoked by the sourced script, not from here
    command() {
        if [ "$1" = "-v" ]; then
            case ":$PII_COV_HIDE_COMMANDS:" in
                *":$2:"*) return 1 ;;
            esac
        fi
        builtin command "$@"
    }
fi
