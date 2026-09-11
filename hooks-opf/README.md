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
  - `PostToolUse` → blocks or warns on tool responses containing PII before the next LLM turn. Claude Code uses a scoped matcher for `Bash`, `Read`, `NotebookRead`, `WebFetch`, `WebSearch`, `Agent`/`Task` (subagent results), `exec_command`, and MCP tools. Codex uses `*` because its tool identifiers vary by runtime. The hook filters returned text, including structural tool output. A live test with `codex-cli 0.154.0` did not invoke the user-level Codex entry for unified shell output.

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

The default `PII_ACTION_MODE=block` rejects input when a span matches the selected `PII_BLOCK_LEVEL`.

Set `PII_ACTION_MODE=warn` to allow the input. The hook then adds a masked detector summary to the agent context. It also tells the agent to check whether each detection is valid. If valid, the agent must avoid repeating the value and use a redacted form. The warning recommends secret rotation or revocation when applicable.

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

Use `PII_ACTION_MODE=warn` for observation or agent-assisted remediation. Use `PII_ACTION_MODE=block` for the hard boundary.

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

2. **Codex trust and tool-path limits.** Codex CLI gates external hooks behind a per-hook trust list. Until you trust each command, Codex registers the hook in `~/.codex/hooks.json` but does not invoke it. Trust lives in `~/.codex/config.toml` under `[hooks.state]`, keyed by `<hooks.json path>:<event>:<group>:<index>`, with `enabled = true` and a `trusted_hash` for the command. Review and trust via `/hooks` in the Codex TUI. Re-running the installer after a command change requires trust again. A live test with `codex-cli 0.154.0` confirmed `UserPromptSubmit` blocking after trust. The same test did not invoke the user-level `PostToolUse` entry for unified shell output. Current Codex coverage is prompt-only. Claude Code runs both hooks.

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
                                        └── configured, but not invoked by codex-cli 0.154.0 in the live test
```

The server is auto-started on first hook call via `uv run`, then stays warm. Health check at `http://127.0.0.1:9123/health`.

For a hard boundary at the file level, see [`agent-seatbelt`](../README.md) (the sandbox in the parent dir).

## Configuration knobs

All env vars override defaults; set them in your shell or the hook's env:

| Var | Default | Purpose |
|---|---|---|
| `PII_BLOCK_LEVEL` | `standard` | tier (off/relaxed/standard/strict) |
| `PII_ALLOW_LABELS` | empty | comma-separated labels to allow within the selected tier |
| `PII_ACTION_MODE` | `block` | `block` to reject input or `warn` to allow input with agent context |
| `PII_SERVER_MODE` | `redact` | `redact` or `openai` |
| `PII_PORT` | `9123` | local server port |
| `PII_SERVER_SCRIPT` | `~/.claude/hooks/pii-server.py` | server script path |
| `PII_SERVER_LOG` | `~/.cache/opf/server.log` | server log path |
| `OPF_CACHE_DIR` | `~/.cache/opf` | model assets cache (server-side) |
| `REDACT_CACHE_DIR` | `~/.cache/redact` | converted Redact assets cache |
| `REDACT_DEVICE` | `auto` | `cuda`, `mps`, or explicit `cpu` |

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

The neural model runs on the selected accelerator. Tokenization, deterministic checks, and span cleanup run on the CPU. Configure the mode with `REDACT_CACHE_DIR`, `REDACT_MIN_SCORE`, `REDACT_BATCH_SIZE`, and `REDACT_MAX_TOKENS`.

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
