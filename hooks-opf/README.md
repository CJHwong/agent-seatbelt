# hooks-opf

Userland PII detector for AI coding agents. Catches secrets and personal data flowing **into** the agent's prompt or **out of** its tool responses, before the LLM ever sees the bytes.

Powered by [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), an int8 ONNX model OpenAI released alongside [their blog post on privacy filtering](https://openai.com/index/introducing-openai-privacy-filter/). Runs entirely locally — the only network call is the one-time model download from Hugging Face on first use.

This is the content-level companion to `agent-seatbelt`'s file-level sandbox. The sandbox stops the agent from reading your secrets; if a secret enters the process anyway (env var, fetched via credential helper, pasted into a prompt), this hook catches it on the way to the LLM.

## What gets installed

- `~/.claude/hooks/pii-check.sh` — the hook binary, called on prompt submit and tool response
- `~/.claude/hooks/pii-server.py` — local HTTP server that loads the ONNX model and returns labeled spans
- For each detected agent, two entries in its hooks config:
  - `UserPromptSubmit` → blocks prompts containing PII before they're sent to the model provider
  - `PostToolUse (*)` → blocks tool responses containing PII before they're fed back to the LLM next turn

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
```

The installer is idempotent — running it again won't duplicate hook entries.

## Requirements

- `jq`, `curl`, `uv` on `PATH`
- macOS or Linux
- Network access on first run (Hugging Face download + `uv` dep resolution)

## Block levels

Tune via `PII_BLOCK_LEVEL`:

| Level | Blocks |
|---|---|
| `off` | nothing |
| `relaxed` | secrets, account numbers |
| `standard` (default) | + emails, phones, addresses |
| `strict` | + names, urls, dates |

Categories outside the blocked tier are still printed to stderr as warnings, so you see what the model flagged without it derailing the request.

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

2. **Codex trust mechanism is required.** Codex CLI gates external hooks behind a per-hook trust list — until you've enabled trust for each command, the hook is registered in `~/.codex/hooks.json` but never invoked. Codex prompts to enable trust on first run; if hooks aren't firing, that's the first thing to check.

3. **Fail-open posture.** The hook returns success (exit 0, empty stdout) on any internal error — server down, jq parse failure, curl timeout. A probabilistic model with a hard fail-closed posture would brick your agent. The tradeoff: missed detections during transient failures are silent. If you need certainty, layer a deterministic regex or block the data source upstream.

## Architecture

```
prompt ──> UserPromptSubmit ──> pii-check.sh --mode prompt ──> pii-server.py
                                       │
                                       └── blocks if any PII at current level
                                       └── prompt sent to Anthropic if clean

tool runs ──> PostToolUse ──> pii-check.sh --mode claude-posttool ──> pii-server.py
                                       │
                                       └── blocks if any PII at current level
                                       └── response fed to LLM next turn if clean
```

The server is auto-started on first hook call via `uv run`, then stays warm. Health check at `http://127.0.0.1:9123/health`.

For a hard boundary at the file level, see [`agent-seatbelt`](../README.md) (the sandbox in the parent dir).

## Configuration knobs

All env vars override defaults; set them in your shell or the hook's env:

| Var | Default | Purpose |
|---|---|---|
| `PII_BLOCK_LEVEL` | `standard` | tier (off/relaxed/standard/strict) |
| `PII_PORT` | `9123` | local server port |
| `PII_SERVER_SCRIPT` | `~/.claude/hooks/pii-server.py` | server script path |
| `PII_SERVER_LOG` | `~/.cache/opf/server.log` | server log path |
| `OPF_CACHE_DIR` | `~/.cache/opf` | model assets cache (server-side) |

## Uninstall

```bash
rm ~/.claude/hooks/pii-check.sh ~/.claude/hooks/pii-server.py
# then edit ~/.claude/settings.json and ~/.codex/hooks.json and remove the entries
```

Model cache lives at `~/.cache/opf/` — remove that too if you want it gone.

## License

None.
