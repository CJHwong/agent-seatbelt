# execlines.awk — which physical lines of a bash script can produce an xtrace
# record. This is the denominator for coverage.sh; getting it wrong is the
# difference between a real number and a demoralizing fake one.
#
# Two facts about bash drive the whole design (both verified empirically on
# bash 5.3 and 3.2 with a PS4 line stamp):
#
#  1. Bash traces *commands*, not lines. A 20-line jq program in single quotes is
#     one command and emits one record. Counting its interior as 19 uncovered
#     lines would peg every action near 40% no matter how well tested it is.
#  2. A multi-line command is reported at its *first* line — EXCEPT a command
#     containing a `(` group (a `$(...)` substitution, a subshell, an array
#     assignment), which is reported at the line where the last group closes
#     (even when the command continues past it). An operator-led continuation
#     (`|| { …; return 1; }`, `| jq …`) is a separate command and is reported
#     at its own line.
#
# So this predicts the first line of each multi-line command, the group close
# line when a `(` is present, and operator-led continuation lines. Quotes,
# backslash continuations, and heredoc bodies are tracked because no
# line-oriented filter can guess them. Modelling the interior more aggressively
# (predicting every continuation line) was measurably worse: it invented misses
# on argument-continuation lines like `--arg id "$fileid"` that bash never
# reports.
#
# Also skipped, because bash emits no record for them:
#   - blanks and comments
#   - pure block syntax (fi / done / esac / else / then / do / braces / ;;),
#     including a closer followed by a case terminator (`fi ;;`) or a heredoc
#     (`done <<EOF`)
#   - case-pattern labels (`foo)` and `foo|bar)`), including a trailing comment
#     or an empty body (`foo) ;;`)
#   - function definition headers (defining is not executing)
#
#   awk -f execlines.awk -v mode=count file   -> the count
#   awk -f execlines.awk -v mode=list  file   -> one executable line number per line
#
# coverage.sh unions this with the lines bash actually traced, so a miss here
# can only ever cost a little accuracy, never silently inflate the percentage.
# SLACKER_COV_AUDIT=1 prints the gap between prediction and observation.
BEGIN { in_s = 0; in_d = 0; prev_cont = 0; hd = ""; n = 0; cmd_start = 0; subst = 0; subst_close = 0; quote_close = 0; is_assign = 0 }
{
  line = $0

  # Heredoc body: part of the command that opened it, so nothing in it is
  # separately executable.
  if (hd != "") {
    t = line; sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
    if (t == hd) hd = ""
    next
  }

  # A line that does not continue a previous one starts a new command. Its
  # reported line is its first line, unless a `(` group closes on a later line
  # (subst_close), or it is a bare assignment whose quoted value spans lines
  # (quote_close), which then wins.
  if (!prev_cont) {
    cmd_start = FNR
    subst_close = 0
    quote_close = 0
    is_assign = ($0 ~ /^[ \t]*[A-Za-z_][A-Za-z0-9_]*=/)
  }

  # --- lexical state for this line ----------------------------------------
  # in_s0/in_d0 are the state at the START of the line (for the operator-led
  # check: an operator inside a quote is not an operator); in_s/in_d end the
  # line still open, which makes this line a continuation of the next.
  # subst counts open `(` groups (substitutions, subshells, array assignments);
  # subst_close records the line where the last one closed.
  in_s0 = in_s; in_d0 = in_d
  i = 1; L = length(line)
  while (i <= L) {
    c = substr(line, i, 1)
    if (in_s) {
      if (c == "'") in_s = 0
    } else if (in_d) {
      if (c == "\\") i++
      else if (c == "\"") in_d = 0
    } else {
      if (c == "\\") i++
      else if (c == "'") in_s = 1
      else if (c == "\"") in_d = 1
      else if (c == "(") { subst++ }
      else if (c == ")") { if (subst > 0) { subst--; if (subst == 0) subst_close = FNR } }
      else if (c == "#" && (i == 1 || substr(line, i - 1, 1) ~ /[ \t;&|(]/)) break
      else if (c == "<" && substr(line, i, 2) == "<<") {
        j = i + 2
        if (substr(line, j, 1) == "-") j++
        if (substr(line, j, 1) == "<") { i = j; continue }   # <<< is a herestring
        q = substr(line, j, 1)
        if (q == "'" || q == "\"") { j++ } else { q = "" }
        w = ""
        while (j <= L) {
          c2 = substr(line, j, 1)
          if (q == "" && c2 !~ /[A-Za-z0-9_]/) break
          if (q != "" && c2 == q) { j++; break }
          w = w c2; j++
        }
        if (w != "") hd = w
        i = j; continue
      }
    }
    i++
  }
  continued = (in_s || in_d) || (line ~ /\\$/)

  # A quote that was already open when the line began and is closed by the end
  # of it: the quoted text spans lines, and this is where it ends. Bash reports a
  # bare assignment of such a value against this line, not the line it started
  # on. Probed on bash 5.3:
  #     mask_jq='        reported at the closing "'"
  #     grep -q '        reported at the opening line (it has a command word)
  if ((in_s0 || in_d0) && !in_s && !in_d) quote_close = FNR

  # --- emit decision --------------------------------------------------------
  out = 0
  t = line; sub(/^[ \t]+/, "", t)
  if (prev_cont) {
    # Continuation of a multi-line command. An operator-led arm is a separate
    # command reported at its own line. The command itself is reported at its
    # substitution close line, else its quoted-value close line when it is a bare
    # assignment, else its first line.
    if (!in_s0 && !in_d0 && t ~ /^(\|\||&&|\|[^|])/) out = FNR
    if (!continued) {
      reported = cmd_start
      if (is_assign && quote_close) reported = quote_close
      if (subst_close) reported = subst_close
      if (reported != out) out = reported
    }
  } else {
    if (!continued) {
      # Single-line command.
      out = FNR
      if (t == "" || substr(t, 1, 1) == "#")                                      out = 0
      else if (t ~ /^(fi|done|esac|else|do|then|\{|\}|\)|;;|;&)([ \t]*;+)?([ \t]*<<[^ \t]*)?[ \t]*$/) out = 0
      else if (t ~ /^\(?[^ \t()|]*(\|[^ \t()|]*)*\)([ \t]*#.*)?([ \t]*;;)?[ \t]*$/) out = 0
      else if (t ~ /^function[ \t]/)                                              out = 0
      else if (t ~ /^[A-Za-z_][A-Za-z0-9_:.-]*[ \t]*\([ \t]*\)[ \t]*\{?[ \t]*$/)   out = 0
    }
    # else: first line of a multi-line command — reported at its end, handled
    # by the prev_cont branch when the command completes.
  }
  if (out) { n++; if (mode == "list") print out }

  prev_cont = continued
}
END { if (mode != "list") print n + 0 }
