# hooks-opf

Userland PII detector for AI coding agents. Catches secrets and personal data flowing **into** the agent's prompt or **out of** its tool responses, before the LLM ever sees the bytes.

The default mode uses Desert Ant Redact on the local GPU. It adds deterministic rules for secrets and private data. OpenAI Privacy Filter remains available as a CPU-only secondary mode.

Both modes run locally. The OpenAI mode downloads its model from Hugging Face on first use. The Redact mode requires a compatible PyTorch cache because the public Redact release does not publish the `redact.pt` checkpoint used by this server.

This is the content-level companion to `agent-seatbelt`'s file-level sandbox. The sandbox stops the agent from reading your secrets; if a secret enters the process anyway (env var, fetched via credential helper, pasted into a prompt), this hook catches it on the way to the LLM.

## What gets installed

- `~/.claude/hooks/pii-check.sh` — the hook binary, called on prompt submit and tool response
- `~/.claude/hooks/pii-server.py` — local HTTP server that loads the selected model and returns labeled spans
- `~/.claude/hooks/redact_server.py` — local Redact model adapter used by `pii-server.py`
- For each detected agent, two entries in its hooks config:
  - `UserPromptSubmit` → blocks or warns on prompts containing PII before they reach the model provider
  - `PostToolUse` → blocks or warns on tool responses containing PII before the next LLM turn. Claude Code uses a scoped matcher for `Bash`, `Read`, `NotebookRead`, `WebFetch`, `WebSearch`, `Agent`/`Task` (subagent results), `exec_command`, and MCP tools. Codex uses `*` because its tool identifiers vary by runtime. The hook filters returned text, including structural tool output.

Supported agents (auto-detected by directory presence):

| Agent | Config file | PostToolUse mode |
|---|---|---|
| Claude Code | `~/.claude/settings.json` | `claude-posttool` |
| Codex | `~/.codex/hooks.json` | `codex-posttool` |

Scripts always land in `~/.claude/hooks/`. Both agents reference the same scripts — no duplication.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-opf/install.sh | bash
```

Flags:

```bash
... | bash -s -- --prompt-only   # skip PostToolUse wiring on both agents
... | bash -s -- --no-codex      # ignore Codex even if ~/.codex/ exists
... | bash -s -- --no-pilot      # skip the pilot warm-up run
```

Before wiring, the installer does a pilot run: it starts the selected server once, smoke-tests it, and leaves it warm. The OpenAI model downloads to `~/.cache/opf/`. The Redact cache must already exist. Pass `--no-pilot` to skip the pilot.

The installer is idempotent. Running it again does not duplicate hook entries. It updates matching entries in place.

## Requirements

- `jq`, `curl`, `uv` on `PATH`
- macOS or Linux
- Network access on first run (Hugging Face download + `uv` dep resolution)

## Block levels

Tune via `PII_BLOCK_LEVEL`. Each level blocks the labels listed below; detected labels outside the blocked set are still printed to stderr as warnings.

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
PII_BLOCK_LEVEL=strict PII_ALLOW_LABELS=private_url,private_date
```

That keeps `strict` blocking enabled for secrets, account numbers, emails, phones, addresses, and names, but allows URLs and dates through as warnings.

When a request is blocked, the hook includes masked snippets in the block message so you can identify what fired without exposing the full value to the agent transcript:

```text
PII in prompt: secret(critical): sk_t...p7dc. Blocked at PII_BLOCK_LEVEL=strict.
```

## Enforcement actions

The default `PII_ACTION_MODE=warn` allows input and adds a masked detector warning to the agent context.

Set `PII_ACTION_MODE=block` to reject input when a span matches the selected `PII_BLOCK_LEVEL`. Set `PII_ACTION_MODE=warn` to allow the input. The hook then adds a masked detector summary to the agent context. It also tells the agent to check whether each detection is valid. If valid, the agent must avoid repeating the value and use a redacted form. The warning recommends secret rotation or revocation when applicable.

Unless `PII_BLOCK_LEVEL=off`, the warning mode reports every detected span. It does not expose the full value. The `PII_BLOCK_LEVEL` setting still controls blocked and warned classifications in stderr. `PII_ALLOW_LABELS` still removes labels from the block set, but warning mode still reports those detector spans.

The hook returns `continue: true` and keeps the current block response unchanged:

```json
{
  "continue": true,
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "PII detector warning: ..."
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

Only works on `UserPromptSubmit`, not on tool responses.

## Tested formats

The fixture at `tests/test-cases.jsonl` covers 25 cases across all label categories. Run against a live server:

```bash
./tests/run-tests.py
python3 tests/test_hook_modes.py
```

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

   Claude Code has a separate PostToolUse caveat. Its `decision: "block"` response leaves the original tool output visible. Use `updatedToolOutput` when the model must not receive the original output. The current block path does not provide that replacement.

3. **Fail-open posture.** The hook returns success (exit 0, empty stdout) on any internal error — server down, jq parse failure, curl timeout. A probabilistic model with a hard fail-closed posture would brick your agent. The tradeoff: missed detections during transient failures are silent. If you need certainty, layer a deterministic regex or block the data source upstream.

## Architecture

```
prompt ──> UserPromptSubmit ──> pii-check.sh --mode prompt ──> pii-server.py
                                       │
                                       └── blocks or warns based on PII_ACTION_MODE
                                       └── prompt sent to Anthropic if clean

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
| `PII_BLOCK_LEVEL` | `standard` | tier (off/relaxed/standard/strict) |
| `PII_ALLOW_LABELS` | empty | comma-separated labels to allow within the selected tier |
| `PII_ACTION_MODE` | `warn` | `warn` to allow input with agent context or `block` to reject input |
| `PII_SERVER_MODE` | `redact` | `redact`, `openai`, or `rules` |
| `PII_PORT` | `9123` | local server port |
| `PII_SERVER_SCRIPT` | `~/.claude/hooks/pii-server.py` | server script path |
| `PII_SERVER_LOG` | `~/.cache/opf/server.log` | server log path |
| `OPF_CACHE_DIR` | `~/.cache/opf` | model assets cache (server-side) |
| `OPF_MAX_TOKENS` | `4096` | OpenAI Privacy Filter request limit; larger requests fail with HTTP 413 |
| `REDACT_CACHE_DIR` | `~/.cache/redact` | converted Redact assets cache |
| `REDACT_DEVICE` | `auto` | `cuda`, `mps`, or explicit `cpu` |
| `REDACT_MAX_TOKENS` | `4096` | maximum tokens in one Redact chunk |
| `REDACT_CHUNK_OVERLAP_TOKENS` | `128` | token overlap between adjacent Redact chunks |
| `REDACT_MAX_INPUT_TOKENS` | `32768` | whole-request token cap; larger requests fail with HTTP 413 before inference |

## Redact GPU mode

The shared server uses Desert Ant Redact by default. It selects NVIDIA `cuda` first, then Apple Metal Performance Shaders (`mps`). It fails if no accelerator exists. Set `REDACT_DEVICE=cpu` only for an explicit CPU run.

Select the default mode on this Apple Silicon machine:

```bash
PII_SERVER_MODE=redact REDACT_DEVICE=mps uv run hooks-opf/pii-server.py --port 9123
```

Check the selected device:

```bash
curl -sS http://127.0.0.1:9123/health
```

The health response identifies the active mode and device:

```json
{"status":"ok","mode":"redact","device":"mps"}
```

Select OpenAI as the secondary mode. Stop the existing server before changing modes on the same port:

```bash
PII_SERVER_MODE=openai uv run hooks-opf/pii-server.py --mode openai --port 9123
```

Select the rules-only mode on a machine with no accelerator, or one too slow for the model:

```bash
PII_SERVER_MODE=rules uv run hooks-opf/pii-server.py --mode rules --port 9123
```

Rules mode runs the deterministic checks only. It loads no checkpoint and uses no accelerator. It reports `{"status":"ok","mode":"rules","device":"cpu"}`. It finds secrets, account numbers, emails, phone numbers, URLs, IP addresses, and dates. It does not find person names or postal addresses, because those need the neural model. On the 25-case fixture it scores 21 of 25; the four misses are the two person-name and two address cases.

Rules mode needs no dependencies. `pii-server.py` imports the standard library alone, so the hook starts it with the system `python3` instead of `uv run`. That avoids resolving the script's declared model dependencies, which include torch. The other two modes still start under `uv run`.

The neural model runs on the selected accelerator. Tokenization, deterministic checks, and span cleanup run on the CPU. Configure the mode with `REDACT_CACHE_DIR`, `REDACT_MIN_SCORE`, `REDACT_BATCH_SIZE`, `REDACT_MAX_TOKENS`, `REDACT_CHUNK_OVERLAP_TOKENS`, and `REDACT_MAX_INPUT_TOKENS`.

### Token limits and long outputs

The Redact checkpoint declares 512 position embeddings. The implementation uses 256-token model windows. It groups those windows into chunks of up to `REDACT_MAX_TOKENS` tokens. Longer input is chunked with `REDACT_CHUNK_OVERLAP_TOKENS` overlap. Deterministic rules scan the full input before model inference. A request above `REDACT_MAX_INPUT_TOKENS` returns HTTP 413 before inference, so the work stays inside the hook's 5 second budget.

The OpenAI Privacy Filter checkpoint declares 131,072 position embeddings. This wrapper keeps `OPF_MAX_TOKENS=4096` as its request limit. OpenAI mode does not chunk input. An oversized request returns HTTP 413, and the hook reports detector failure instead of silently allowing it.

Review the [Redact release](https://huggingface.co/desert-ant-labs/redact/resolve/v0.4.0/README.md) and its [source-available license](https://license.desertant.com/1.0) before distribution. The published release contains Core ML and TFLite assets. Create or provide the PyTorch cache separately.

Measure resource use on the final holdout corpus:

```bash
uv run hooks-opf/tests/run-resource-comparison.py --mode local
uv run hooks-opf/tests/run-resource-comparison.py --mode openai
```

The report includes process CPU time, wall latency, peak resident memory, and accelerator allocation. PyTorch MPS does not expose a reliable GPU utilization percentage.

## Expanded comparison corpus

The [false-positive corpus](tests/false-positive-cases.jsonl) has 102 cases. It contains 50 strict clean cases, 32 required detections, and 20 policy-ambiguous cases.

Strict clean cases contain no intended PII. Any returned span counts as a false positive. Required detections check label recall. Ambiguous cases cover public examples, test values, and business data. They are reported but not scored as false positives.

The [comparison runner](tests/run-comparison.py) accepts any server with the shared `POST /` contract:

```bash
uv run hooks-opf/tests/run-comparison.py \
  --server redact=http://127.0.0.1:9124 \
  --server openai=http://127.0.0.1:9123
```

Exclude labels that a model documents as unsupported. For example, Rampart does not model dates or catch-all secrets:

```bash
--unsupported rampart=private_date,secret
```

## Uninstall

```bash
rm ~/.claude/hooks/pii-check.sh ~/.claude/hooks/pii-server.py ~/.claude/hooks/redact_server.py
# then edit ~/.claude/settings.json and ~/.codex/hooks.json and remove the entries
```

Model cache lives at `~/.cache/opf/` — remove that too if you want it gone.

## License

None.
