# hooks-pii evaluation

## Tested formats

The fixture at `tests/test-cases.jsonl` covers 25 cases across all label categories. Run against a live server:

```bash
./evals/run-tests.py                                          # detector accuracy, needs a live server
python3 -m unittest discover -s tests -p 'test_hook_*.py'     # the shell hook, no model needed
python3 -m unittest discover -s tests -p 'test_install.py'    # the installer, no model needed
bash tests/coverage.sh                                        # line coverage for the shipped bash
uv run --with coverage python -m coverage run --branch --source=. \
    -m unittest discover -s tests -p 'test_*.py'              # branch coverage, Python side
uv run --with coverage python -m coverage report -m
```

The hook tests stub the detector with a throwaway HTTP server, and the installer tests run the real `install.sh` against a temporary `HOME` with the source pointed at this checkout, so both suites run offline in seconds and neither touches your `~/.claude` or `~/.codex`.

### The canary suite

`tests/test_hook_canaries.py` asks a different question from every other test here. The rest assert a response: the fields, the shape, the wording. This one asserts the **effect**, by running the real `claude` CLI against a scripted API that serves one fixed tool call, and then checking whether the command the hook refused actually ran.

A response test cannot catch a control that is wired up wrong. A hook has two ways to look right and stop nothing: the runtime can drop its decision, and the wrapper that invokes it can swallow the decision before the runtime sees it. The second kind is easy to write and fails silently. Both kinds pass every shape assertion.

The scripted API is what makes this a test. With a real model in the loop, the model decides whether to call the tool at all, and it will refuse a prompt that reads like a probe, so an absent side effect proves nothing. `tests/stub_anthropic_api.py` answers instead.

Every control is measured twice inside the same test, once off and once on. Kept apart, a block test passes whenever the harness cannot produce the effect at all, which is how a broken harness reads as a working control.

The suite passes `--allowedTools Bash` to the CLI. That is not decoration: without it the scripted tool call runs only where the machine already trusts the workspace. The suite passed on macOS and failed on Linux with the same CLI version until that flag went in.

CI installs the current CLI release on purpose, because noticing a hook contract change is what this suite is for. It skips wherever `claude` is absent, and the suite table reports the skip count, so a green run cannot quietly mean "not checked".

Measured on Claude Code 2.1.276: PreToolUse refuses on a nested `permissionDecision`, on a top-level `decision`, and on exit code 2. The hook emits the nested form because Claude Code documents it, not because the other two fail. One consequence is worth knowing: Claude Code asks the provider for a session title before any hook runs, and that call carries the prompt text, so a blocked prompt still reaches the provider once. The turn never starts, which is the guarantee the hook offers.

`coverage.sh` covers every shipped bash file, `hook/check.sh` and `install.sh`, on lines only, and applies `PII_COV_FLOOR` to **each file** rather than to the total: a total lets one file improve while another regresses and still passes, which is the opposite of a ratchet. Set `PII_COV_FLOOR=100` in CI, and `PII_COV_DETAIL=1` to list the uncovered lines.

Measure branch coverage, not only lines. Lines reached hide a guard whose false arm no test ever takes: a rule can be 100% covered and still be broken for the most common input of its kind, because a guard that always evaluates true is still a reached line. That is not hypothetical. It happened here, to the phone rule, which was fully covered and dropped every phone number at the end of a sentence. The column to watch is `BrPart`.

Current pass rate against `openai/privacy-filter` (int8 quantized): **24/25**.

| Category | Pass |
|---|---|
| `private_person` | 2/2 |
| `private_address` | 2/2 |
| `private_email` | 2/2 |
| `private_phone` | 2/2 |
| `private_url` | 2/2 |
| `private_date` | 2/2 |
| `account_number` | 2/2 |
| `secret` | 10/11 |

### Known gap: bare AWS access keys in prose

`AKIAIOSFODNN7EXAMPLE` floating as a bare token in narrative text returns zero spans. The same key inside an `export AWS_ACCESS_KEY_ID=...` shell-export form, or paired with a realistic-looking `AWS_SECRET_ACCESS_KEY`, is correctly labeled `secret`. The realistic leak path — `cat .env`, `aws configure list`, `printenv | grep AWS` — carries the env-var context and is caught. Bare-token-in-prose is rare in actual tool output.

If your threat model includes bare AWS keys in narrative text, layer a 3-line `AKIA[0-9A-Z]{16}` regex check ahead of this hook.

## Expanded comparison corpus

The [false-positive corpus](../tests/false-positive-cases.jsonl) has 102 cases. It contains 48 strict clean cases, 33 required detections, and 21 policy-ambiguous cases. Only the required detections are enforced: `run-comparison.py` checks `expected` for a `positive` case and reports an ambiguous case's spans as an informational flag, so an `expected` list on an ambiguous case documents a policy call rather than gating it.

Strict clean cases contain no intended PII. Any returned span counts as a false positive. Required detections check label recall. Ambiguous cases cover public examples, test values, and business data. They are reported but not scored as false positives.

The [comparison runner](../evals/run-comparison.py) accepts any server with the shared `POST /` contract:

```bash
uv run hooks-pii/evals/run-comparison.py \
  --server redact=http://127.0.0.1:9124 \
  --server openai=http://127.0.0.1:9123
```

Exclude labels that a model documents as unsupported. For example, Rampart does not model dates or catch-all secrets:

```bash
--unsupported rampart=private_date,secret
```
