# hooks-pii

Userland PII detector for AI coding agents. Catches secrets and personal data flowing **into** the agent's prompt or **out of** its tool responses, before the LLM ever sees the bytes.

The default mode uses [Desert Ant Redact](https://desertant.com) on the local GPU. It adds deterministic rules for secrets and private data. OpenAI Privacy Filter remains available as a CPU-only secondary mode.

Both modes run locally. The OpenAI mode downloads its model from Hugging Face on first use. The Redact mode requires a compatible PyTorch cache because the public Redact release does not publish the `redact.pt` checkpoint used by this server.

This is the content-level companion to `agent-seatbelt`'s file-level sandbox. The sandbox stops the agent from reading your secrets; if a secret enters the process anyway (env var, fetched via credential helper, pasted into a prompt), this hook catches it on the way to the LLM.

## What gets installed

- `~/.claude/hooks/pii-check.sh` — the hook binary, called on prompt submit and tool response
- `~/.claude/hooks/pii-server.py` — local HTTP server that loads the selected model and returns labeled spans
- `~/.claude/hooks/pii_redact_torch.py` — local Redact model adapter used by `pii-server.py`
- `~/.claude/hooks/pii_rules_native.abi3.so` — the native rules engine, on macOS arm64 and Linux x86_64 only. See [Native rules engine](#native-rules-engine)
- For each detected agent, two entries in its hooks config:
  - `UserPromptSubmit` → blocks or warns on prompts containing PII before they reach the model provider
  - `PreToolUse` → blocks or warns on a tool call's **input** before it runs. This is the only
    point where the value has not left the machine: blocking here stops the command, where
    blocking on tool output is a report after the fact. Claude Code uses a scoped matcher
    for `Bash`, `exec_command`, `WebFetch`, `WebSearch`, `Agent`/`Task` and MCP tools. `Bash`
    covers `wget`, `curl`, `scp` and every other command, because the matcher names tools rather
    than binaries. `Read` and `NotebookRead` are absent because their input is a path, and
    `Edit`/`Write` because scanning what the agent just wrote is wasted work. Codex uses `*`,
    because it resolves `Edit`, `Write` and `apply_patch` to one tool name and renames its shell
    tool across versions, so a scoped pattern there would be unstable. The detector only sees
    tool input when `--prompt-only` was not passed, the same flag that skips `PostToolUse`.
  - A timed-out hook **fails open** on PreToolUse: the call continues through the normal
    permission flow, so a stalled scanner is not a gate. The hook's own POST budget is 5
    seconds and the entry allows 20, so the scanner has to finish inside that. If you
    raise `REDACT_MAX_INPUT_TOKENS` or point `PII_PORT` at a slower host, check the two
    numbers still fit.
  - `PostToolUse` → blocks or warns on tool responses containing PII before the next LLM turn. Claude Code uses a scoped matcher for `Bash`, `Read`, `NotebookRead`, `WebFetch`, `WebSearch`, `Agent`/`Task` (subagent results), `exec_command`, and MCP tools. Codex uses `*` because its tool identifiers vary by runtime. The hook filters returned text, including structural tool output.

Supported agents (auto-detected by directory presence):

| Agent | Config file | PostToolUse mode | PreToolUse mode |
|---|---|---|---|
| Claude Code | `~/.claude/settings.json` | `claude-posttool` | `claude-pretool` |
| Codex | `~/.codex/hooks.json` | `codex-posttool` | `codex-pretool` |

Scripts always land in `~/.claude/hooks/`. Both agents reference the same scripts — no duplication.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-pii/install.sh | bash
```

Flags:

```bash
... | bash -s -- --prompt-only   # skip both tool hooks, keeping the prompt hook alone
... | bash -s -- --no-codex      # ignore Codex even if ~/.codex/ exists
... | bash -s -- --no-pilot      # skip the pilot warm-up run
```

Before wiring, the installer does a pilot run: it starts the selected server once, smoke-tests it, and leaves it warm. The OpenAI model downloads to `~/.cache/opf/`. The Redact cache must already exist. Pass `--no-pilot` to skip the pilot.

The installer is idempotent. Running it again does not duplicate hook entries. It updates matching entries in place.

## Requirements

- `jq`, `curl`, `uv` on `PATH`
- macOS or Linux
- Network access on first run (Hugging Face download + `uv` dep resolution)

## Levels

Tune via `PII_LEVEL`. The level selects which labels the hook acts on. Block mode rejects a selected span. Warn mode reports a selected span to the agent. A label below the level is printed to stderr only, so the agent never sees it.

`PII_BLOCK_LEVEL` is the former name. The hook still reads it when `PII_LEVEL` is unset.

| Level | Blocked labels |
|---|---|
| `off` | nothing |
| `relaxed` | `secret`, `account_number` |
| `standard` (default) | `secret`, `account_number`, `private_email`, `private_phone`, `private_address` |
| `strict` | `secret`, `account_number`, `private_email`, `private_phone`, `private_address`, `private_person`, `private_url`, `private_date` |

`PII_ALLOW_LABELS` accepts these label names:

| Label | Tier | Typical data |
|---|---|---|
| `secret` | critical | API keys, tokens, passwords, private keys |
| `account_number` | critical | bank accounts, card numbers, account IDs |
| `private_email` | moderate | personal or private email addresses |
| `private_phone` | moderate | phone numbers |
| `private_address` | moderate | street addresses |
| `private_person` | low | people's names |
| `private_url` | low | private or internal URLs |
| `private_date` | low | personal or sensitive dates |

To carve out specific categories from a tier, set `PII_ALLOW_LABELS` to a comma-separated list:

```bash
PII_LEVEL=strict PII_ALLOW_LABELS=private_url,private_date
```

That keeps `strict` enabled for secrets, account numbers, emails, phones, addresses, and names, but allows URLs and dates through. An allowed label goes to stderr only, in both action modes.

When a request is blocked, the hook includes masked snippets in the block message so you can identify what fired without exposing the full value to the agent transcript:

```text
PII in prompt: secret(critical): sk...dc. Blocked at PII_LEVEL=strict.
```

## Enforcement actions

The default `PII_ACTION_MODE=warn` allows input and adds the masked detector summary to both `systemMessage` and `hookSpecificOutput.additionalContext`.

Set `PII_ACTION_MODE=block` to reject input when a span matches the selected `PII_LEVEL`. Set `PII_ACTION_MODE=warn` to allow the input. The hook then adds a masked detector summary to the agent context. It also tells the agent to check whether each detection is valid. If valid, the agent must avoid repeating the value and use a redacted form. The warning recommends secret rotation or revocation when applicable.

A skipped scan says so. Each trigger is a standing condition, not a one-off: a missing `jq`
or a misspelled `PII_ACTION_MODE` stays that way, so the next request is unscanned too. The
message carries the cause, the fix, and the fact that the scanner stays off until someone acts,
which is what an agent needs in order to tell the user that a check they rely on is not running.
It does not block, even in block mode: a missing tool is an operational fault rather than
evidence about the content, and blocking would take the agent down with no way for it to clear.

A detector failure reads the same way. The detector can be unreachable, stuck on the wrong
mode, or answering with a shape the hook cannot use, and each one names itself in the message
and writes one line to `~/.cache/pii/pii-skips.log`. An oversized input does too. Both leave
the request unscanned, so both are recorded where a person can read them without the agent.

Both action modes honour `PII_LEVEL`. Warn mode reports the same spans that block mode would reject. It does not expose the full value. A span below the level goes to stderr as `PII below level:`, and the agent never receives it. `PII_ALLOW_LABELS` removes a label in both modes.

At `PII_LEVEL=relaxed` with `PII_ACTION_MODE=warn`, a prompt carrying only a name and a phone number produces no agent-visible warning. Raise the level to see those categories again.

The hook returns `continue: true` and keeps the current block response unchanged:

```json
{
  "continue": true,
  "systemMessage": "PII detector warning: possible sensitive data was identified in the user prompt: secret(critical): [masked].",
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "PII detector warning: ... masked findings ..."
  }
}
```

Use `PII_ACTION_MODE=warn` for gradual adoption, observation, or agent-assisted remediation. Use `PII_ACTION_MODE=block` for the hard boundary.

### Planned mask mode

Mask mode is not implemented. Do not set `PII_ACTION_MODE=mask`.

The planned behavior would replace detected values with fixed markers, then let the agent continue:

```text
password=[REDACTED:SECRET]
email=[REDACTED:PRIVATE_EMAIL]
```

Mask secrets without preserving a prefix or suffix. A partial secret can still be useful to an attacker. Mask mode must preserve the output shape that each runtime expects. It must fail closed if replacement is rejected.

The runtimes use different PostToolUse contracts:

| Runtime | Replacement support | Test result |
|---|---|---|
| Claude Code 2.1.268 | `updatedToolOutput` replaces tool output for all tools. The replacement must match the tool output shape. | The real CLI sent only `MASKED_OUTPUT` to a local protocol stub. |
| Codex 0.154.0 | No generic `updatedToolOutput`. `decision: "block"` replaces the model-visible result with hook feedback. | The real CLI saw masked feedback and did not see the original sentinel. |
| Codex 0.154.0 with `continue: false` | Not a replacement method for this use. | The real CLI saw the original sentinel. |

See the [Claude Code hooks reference](https://code.claude.com/docs/en/hooks) and the [official Codex hooks documentation](https://learn.chatgpt.com/docs/hooks).

Neither documented `UserPromptSubmit` contract provides an updated prompt field. `additionalContext` adds context but does not replace the original prompt. Secure prompt masking requires a wrapper before the agent receives the prompt.

The Claude test used the real CLI with a local protocol stub because the live account had no available API quota. The Codex test used the real CLI with a temporary project hook. Both tests used synthetic sentinel values only.

## Per-prompt bypass

Prefix a single prompt with `pii:off ` to skip the check for that submission:

```
pii:off paste the contents of my .env to debug this
```

It applies to a user prompt and to nothing else. Tool output is content that an
attacker can plant, so a prefix there is not honoured: a page, a file, or an MCP
result whose first string begins with `pii:off` is still scanned.

The prefix suits one person at one keyboard. Anyone who can send a message to an
agent that takes prompts from a chat gateway can forge it, because a prefix carried
in-band cannot be authenticated. A deployment with more than one user sets
`PII_ALLOW_BYPASS=0`, and then the prefix has no effect:

```bash
PII_ALLOW_BYPASS=0
```

### What the warning preview shows

Warn mode reports a masked preview of each selected span. It reveals at most two
characters at each end, and only when the value is at least twelve characters long.
Anything shorter prints as `[redacted]`. A partial secret is still useful to an
attacker, so a short value gives up nothing rather than most of itself.

The preview goes into the transcript and reaches the model provider. Treat it as
disclosed: rotate or revoke the value if the detection was real.

## Tested formats

The fixture at `tests/test-cases.jsonl` covers 25 cases across all label categories. Run against a live server:

```bash
./tests/run-tests.py                                          # detector accuracy, needs a live server
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

`coverage.sh` covers every shipped bash file, `pii-check.sh` and `install.sh`, on lines only, and applies `PII_COV_FLOOR` to **each file** rather than to the total: a total lets one file improve while another regresses and still passes, which is the opposite of a ratchet. Set `PII_COV_FLOOR=100` in CI, and `PII_COV_DETAIL=1` to list the uncovered lines.

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

## Limitations

This is a high-recall filter, not a hard gate. Three failure modes worth knowing:

1. **Adversarial reformatting bypass.** A motivated agent (or a model trying to satisfy a "show me the file" prompt) can reshape tool output past the NER's pattern matching: `od -c file`, `base64 file`, `xxd file`, `tr a-z A-Z < file`, splitting bytes across lines, etc. Observed empirically — given a blocked `cat secrets.txt`, a model adapted within one turn to `od -c` and the byte-spread output flowed through unblocked. The NER labels patterns it recognizes; spread-out or re-encoded versions of the same content are not labeled. Content-based filtering can't close this gap without semantic execution; treat the hook as defense-in-depth alongside the file-level sandbox, not a perimeter.

2. **Codex trust requirement.** Codex CLI gates external hooks behind a per-hook trust list. Until you trust each command, Codex registers the hook in `~/.codex/hooks.json` but does not invoke it. Trust lives in `~/.codex/config.toml` under `[hooks.state]`, keyed by `<hooks.json path>:<event>:<group>:<index>`, with `enabled = true` and a `trusted_hash` for the command. Review and trust via `/hooks` in the Codex TUI. Re-running the installer after a command change requires trust again. A fresh live test with `codex-cli 0.154.0` confirmed both `UserPromptSubmit` and `PostToolUse` blocking for unified shell output. Warning mode also displayed the masked `systemMessage` and continued the turn. Claude Code runs both hooks without a trust step.

   Two constraints apply to a Codex PreToolUse deny. Its hook output schema sets
   `additionalProperties: false`, so one unexpected key discards the entire reply, and a
   discarded reply does not block. And `ask`, `continue: false`, `stopReason` and
   `suppressOutput` are parsed but unsupported: returning one marks the run `Failed` while
   the tool call proceeds. The `deny` shape this hook emits carries only keys Codex knows.
   A `deny` with a non-empty reason is `Blocked`; anything malformed is not.

   Trust is recorded against the command string, and this hook's path is fixed, so a content
   update does not force a re-trust. Worth knowing for what it implies: existing trust then
   covers future changes to the script's content. Claude Code behaves the same way, so it is
   not a Codex-specific weakness, but it is why a hook's content deserves the same review as
   its wiring.

   Claude Code has a separate PostToolUse caveat. Its `decision: "block"` response leaves the original tool output visible. Use `updatedToolOutput` when the model must not receive the original output. The current block path does not provide that replacement.

3. **Fail-open posture.** The hook returns success (exit 0, empty stdout) on any internal error — server down, jq parse failure, curl timeout. A probabilistic model with a hard fail-closed posture would brick your agent. The tradeoff: missed detections during transient failures are silent. If you need certainty, layer a deterministic regex or block the data source upstream.

## Architecture

```
prompt ──> UserPromptSubmit ──> pii-check.sh --mode prompt ──> pii-server.py
                                       │
                                       └── blocks or warns based on PII_ACTION_MODE
                                       └── prompt sent to Anthropic if clean

Claude tool call ──> PreToolUse ──> pii-check.sh --mode claude-pretool ──> pii-server.py
Claude tool runs ──> PostToolUse ──> pii-check.sh --mode claude-posttool ──> pii-server.py
                                       │
                                       └── blocks or warns based on PII_ACTION_MODE
                                       └── response fed to LLM next turn if clean

Codex tool runs ──> PostToolUse ──> pii-check.sh --mode codex-posttool ──> pii-server.py
                                        │
                                        └── blocks or warns based on PII_ACTION_MODE
```

The server is auto-started on first hook call via `uv run`, then stays warm. Health check at `http://127.0.0.1:9123/health`.

For a hard boundary at the file level, see [`agent-seatbelt`](../README.md) (the sandbox in the parent dir).

## Configuration knobs

All env vars override defaults; set them in your shell or the hook's env:

| Var | Default | Purpose |
|---|---|---|
| `PII_LEVEL` | `standard` | tier (off/relaxed/standard/strict); `PII_BLOCK_LEVEL` is the legacy name |
| `PII_ALLOW_LABELS` | empty | comma-separated labels to allow within the selected tier |
| `PII_ALLOW_BYPASS` | `1` | `1` enables the `pii:off` prompt prefix; `0` disables it, for deployments where more than one person can reach the agent |
| `PII_ACTION_MODE` | `warn` | `warn` to allow input with agent context or `block` to reject input |
| `PII_SERVER_MODE` | `redact` | `redact` (LiteRT graph), `redact-torch` (checkpoint), `openai`, or `rules` |
| `PII_PORT` | `9123` | local server port |
| `PII_SERVER_SCRIPT` | `~/.claude/hooks/pii-server.py` | server script path |
| `PII_SERVER_LOG` | `~/.cache/pii/server.log` | server log path |
| `PII_MAX_BODY_BYTES` | `2097152` | largest request body either server reads (2 MiB); a larger declared length is answered with HTTP 413 before the body is read. A client still streaming when the answer is sent can see a broken pipe instead of the 413 |
| `OPF_CACHE_DIR` | `~/.cache/opf` | OpenAI Privacy Filter assets, this backend only |
| `OPF_MAX_TOKENS` | `1024` | OpenAI Privacy Filter request limit; larger requests fail with HTTP 413 |
| `REDACT_CACHE_DIR` | `~/.cache/redact` | converted Redact assets cache |
| `REDACT_DEVICE` | `auto` | `redact-torch` only: `cuda`, `mps`, or explicit `cpu`. `redact` is always CPU |
| `REDACT_MAX_TOKENS` | `4096` | maximum tokens in one Redact chunk |
| `REDACT_CHUNK_OVERLAP_TOKENS` | `128` | token overlap between adjacent Redact chunks |
| `REDACT_MAX_INPUT_TOKENS` | `32768` | whole-request token cap; larger requests fail with HTTP 413 before inference |

## Redact modes

Redact is the default detector and it has two backends, which differ in where the
weights come from and in what they need to run.

`redact` runs the published LiteRT graph. It downloads `redact.tflite`,
`config.json` and `tokenizer.json` from the Redact release at v0.4.0 on first use,
needs no accelerator, and runs on a machine with nothing prepared. It is what a
fresh install gets.

`redact-torch` runs the PyTorch checkpoint instead. It needs `redact.pt` already
in the cache, which the public release no longer publishes, so it is for a host
that has one. Where it can run it is 2 to 3 times faster: it batches eight windows
into one forward pass, which `redact` cannot do because its graph is fixed at one
window per call, and it can use a GPU where `redact` has no working accelerator.

The two agree: on the repo's own 440-case corpora they produce identical spans,
and on six long documents they agree on five exactly and differ by one span on the
sixth.

Select the default:

```bash
PII_SERVER_MODE=redact uv run hooks-pii/pii-server.py --port 9123
```

Select the checkpoint backend, which needs `~/.cache/redact/redact.pt` and an
accelerator. It selects NVIDIA `cuda` first, then Apple Metal Performance Shaders
(`mps`), and fails if no accelerator exists:

```bash
PII_SERVER_MODE=redact-torch REDACT_DEVICE=mps uv run hooks-pii/pii-server.py --port 9123
```

Check the selected device:

```bash
curl -sS http://127.0.0.1:9123/health
```

The health response identifies the active mode and device:

```json
{"status":"ok","mode":"redact","device":"cpu"}
```

Select OpenAI as the secondary mode. Stop the existing server before changing modes on the same port:

```bash
PII_SERVER_MODE=openai uv run hooks-pii/pii-server.py --mode openai --port 9123
```

Select the rules-only mode on a machine with no accelerator, or one too slow for the model:

```bash
PII_SERVER_MODE=rules uv run hooks-pii/pii-server.py --mode rules --port 9123
```

Rules mode runs the deterministic checks only. It loads no checkpoint and uses no accelerator. It reports `{"status":"ok","mode":"rules","device":"cpu"}`. It finds secrets, account numbers, emails, phone numbers, URLs, IP addresses, and dates. It does not find person names or postal addresses, because those need the neural model. On the 25-case fixture it scores 21 of 25; the four misses are the two person-name and two address cases.

Rules mode needs no dependencies. `pii-server.py` imports the standard library alone, so the hook starts it with the system `python3` instead of `uv run`. That avoids resolving the script's declared model dependencies, which include torch. The other two modes still start under `uv run`.

In `redact-torch` the neural model runs on the selected accelerator. Tokenization, deterministic checks, and span cleanup run on the CPU, and so does everything in `redact`, whose graph is CPU only. Configure the mode with `REDACT_CACHE_DIR`, `REDACT_MIN_SCORE`, `REDACT_BATCH_SIZE` (`redact-torch` only, because the LiteRT graph cannot batch), `REDACT_MAX_TOKENS`, `REDACT_CHUNK_OVERLAP_TOKENS`, and `REDACT_MAX_INPUT_TOKENS`.

### Native rules engine

The rules are Python regular expressions. On a slow CPU, Python `re` takes too long on a large tool output: 105 ms on a 12 KB input on a 2015 Celeron N3050. The native engine matches the same patterns in compiled code. `pii_rules.py` passes its own pattern strings to it, so the rules still live in one file.

It makes two changes:

- PCRE2 compiles each pattern to machine code. PCRE2 supports the lookarounds and the conditional group that the rules use.
- Each pattern names the literals it cannot match without, in `RULE_KEYWORDS`. One Aho-Corasick pass over the text finds the literals that are present. A pattern whose literals are all absent does not run. The Python engine uses the same gates.

Measured through the server on 1,000 real transcript inputs per machine. The time is the server's `processing_ms`, rules mode:

| Machine | p99 input | Python `re` at p99 | Native at p99 | Inputs under 10 ms, native |
|---|---|---|---|---|
| Apple M1 Pro | 35 KB | 16 ms | 3 ms | 99.8% |
| Celeron N3050 | 14 KB | 31 ms | 9 ms | 99.2% |

Both engines returned the same spans for all 2,000 inputs.

The native engine returns the same spans as `re`. PCRE2 and `re` differ in three places, and each is handled:

- **Word characters.** PCRE2 counts combining marks and connector punctuation as word characters for `\w` and `\b`, and `re` does not. Each pattern is compiled twice: once as written, and once with `\w` and `\b` spelled the way `re` reads them. A text holding one of those characters uses the spelled form, which is half as fast.
- **Unicode versions.** Python 3.11 knows Unicode 14, and PCRE2 knows Unicode 16. When the versions differ, `pii_rules` runs each character class over every code point in both engines at start. A text holding a character they place differently goes to `re`. This takes 0.2 s on an M1, and it is skipped when the versions match.
- **Case folding.** Four characters match an ASCII letter when case is ignored (`İ`, `ı`, `ſ`, and the Kelvin sign), and an ASCII keyword search cannot see them. A text holding one goes to `re`. So does a text with a lone surrogate, which cannot be passed to Rust.

The test suite compares the two engines on every case text in the repository, and on every code point for each character class.

The installer downloads the build for the platform from the latest GitHub release. On any other platform, or when the download fails, the rules run on the Python engine. The spans are the same, but the scan is slower. The server log names the engine at start:

```
[rules] rules engine: native
[rules] rules engine: python (No module named 'pii_rules_native')
```

To build it yourself, install Rust and run:

```bash
cd hooks-pii/native && cargo build --release
cp target/release/libpii_rules_native.dylib ../pii_rules_native.abi3.so   # macOS
cp target/release/libpii_rules_native.so ../pii_rules_native.abi3.so      # Linux
```

The file is built against the Python stable ABI, so one build loads on CPython 3.9 and later. Restart the server after you replace the file.

### Provider token patterns

`pii_secret_patterns.py` holds 71 provider token rules from [gitleaks](https://github.com/gitleaks/gitleaks) (MIT). They are the gitleaks rules that detect a secret type on GitHub's [secret scanning list](https://docs.github.com/en/code-security/secret-scanning/introduction/supported-secret-scanning-patterns). `tests/github-secret-types.tsv` lists each GitHub type and the rule that covers it. 82 of the 470 types have a rule. A rule reports a match only when the secret passes the gitleaks entropy floor and allowlist, and only when a gitleaks keyword is in the text.

The port leaves out the gitleaks rules that pair a keyword with any string of the right length and have no entropy floor. gitleaks `adafruit-api-key`, for one, flags `adafruit_feed = "temperature-sensor-living-room-1"`.

### Token limits and long outputs

The Redact checkpoint declares 512 position embeddings. The implementation uses 256-token model windows holding 254 content tokens, advanced with a step of 190, so adjacent windows share 64 tokens. The code names those `CONTENT_WINDOW_LENGTH`, `WINDOW_OVERLAP` and `WINDOW_STEP`, and `WINDOW_STEP` is derived from the other two rather than written out. It groups those windows into chunks of up to `REDACT_MAX_TOKENS` tokens. Longer input is chunked with `REDACT_CHUNK_OVERLAP_TOKENS` overlap, which is separate from the window overlap. Deterministic rules scan the full input before model inference. A request above `REDACT_MAX_INPUT_TOKENS` returns HTTP 413 before inference, but only after the whole text has been tokenized, so the rejection costs memory in proportion to the body rather than to the cap: a 4 MB body was measured taking 2.5 s and about 640 MB of growth to reach its 413, and a 1 MB body reached it in 0.3 s without the peak moving. Note that the cap bounds tokens, not the number of forward passes. A request at the cap is subdivided into roughly 180 model windows, so it is far more work than the token count suggests.

One asymmetry is worth knowing and is left in place deliberately. A token that sits on a window seam is covered by two windows, and the aggregation takes the maximum across them and then renormalizes, which can only lower that token's score. Seams fall every 190 tokens, so an entity landing on one scores slightly below the same entity elsewhere. The bias is small and one-directional; it is recorded in the code rather than corrected.

The OpenAI Privacy Filter checkpoint declares 131,072 position embeddings. This wrapper keeps `OPF_MAX_TOKENS=1024` as its request limit, which costs 2.28 s with one intra-op thread and 0.70 s with four. The hook's POST budget is 5 seconds and must also carry the round trip and the JSON parse, so the previous 4096 default accepted work that took 13 seconds on one thread and could never have fit. Raise it with `OPF_MAX_TOKENS` on a machine that can afford it: 4096 tokens takes about 3.9 s on four threads.

OpenAI mode does not chunk, so an oversized request returns HTTP 413 before any inference. The hook then names the size as the cause rather than blaming the detector, and tells the agent to raise the limit. In warn mode the content is allowed through unscanned, which is stated in the message; in block mode it is blocked, because a value that cannot be scanned cannot be cleared.

Review the [Redact release](https://huggingface.co/desert-ant-labs/redact/resolve/v0.4.0/README.md) and its [source-available license](https://license.desertant.com/1.0) before distribution. The release publishes `redact.tflite` and a compiled Core ML model. It does not publish `redact.pt`: every published revision is missing it, and the one commit that still lists it answers 403 from the storage layer. So `redact-torch` needs a checkpoint that was prepared separately, which is why `redact` is the default.

Measure resource use on the final holdout corpus:

```bash
uv run hooks-pii/tests/run-resource-comparison.py --mode local
uv run hooks-pii/tests/run-resource-comparison.py --mode openai
```

The report includes process CPU time, wall latency, peak resident memory, and accelerator allocation. PyTorch MPS does not expose a reliable GPU utilization percentage.

## Expanded comparison corpus

The [false-positive corpus](tests/false-positive-cases.jsonl) has 102 cases. It contains 48 strict clean cases, 33 required detections, and 21 policy-ambiguous cases. Only the required detections are enforced: `run-comparison.py` checks `expected` for a `positive` case and reports an ambiguous case's spans as an informational flag, so an `expected` list on an ambiguous case documents a policy call rather than gating it.

Strict clean cases contain no intended PII. Any returned span counts as a false positive. Required detections check label recall. Ambiguous cases cover public examples, test values, and business data. They are reported but not scored as false positives.

The [comparison runner](tests/run-comparison.py) accepts any server with the shared `POST /` contract:

```bash
uv run hooks-pii/tests/run-comparison.py \
  --server redact=http://127.0.0.1:9124 \
  --server openai=http://127.0.0.1:9123
```

Exclude labels that a model documents as unsupported. For example, Rampart does not model dates or catch-all secrets:

```bash
--unsupported rampart=private_date,secret
```

## Uninstall

```bash
rm ~/.claude/hooks/pii-check.sh ~/.claude/hooks/pii-server.py ~/.claude/hooks/pii_redact_torch.py
rm -f ~/.claude/hooks/pii_rules_native.abi3.so
# then edit ~/.claude/settings.json and ~/.codex/hooks.json and remove the entries
```

Runtime files live under `~/.cache/pii/` (the server log and the skip events).
Each backend keeps its model assets in its own cache: `~/.cache/opf/` for the
OpenAI Privacy Filter, `~/.cache/redact/` for Redact. Remove those too if you
want them gone.

An install made before the suite had its own directory kept `server.log` and
`pii-skips.log` under `~/.cache/opf/`. Move them to `~/.cache/pii/` by hand if
you want to keep the history; a stale one left behind is simply not read.

## License

None.
