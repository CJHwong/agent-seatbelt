# hooks-pii

Userland PII detector for AI coding agents. Catches secrets and personal data flowing **into** the agent's prompt or **out of** its tool responses, before the LLM ever sees the bytes.

The default mode uses [Desert Ant Redact](https://desertant.com) on the local GPU. It adds deterministic rules for secrets and private data. OpenAI Privacy Filter remains available as a CPU-only secondary mode.

Both modes run locally. The OpenAI mode downloads its model from Hugging Face on first use. The Redact mode requires a compatible PyTorch cache because the public Redact release does not publish the `redact.pt` checkpoint used by this server.

This is the content-level companion to `agent-seatbelt`'s file-level sandbox. The sandbox stops the agent from reading your secrets; if a secret enters the process anyway (env var, fetched via credential helper, pasted into a prompt), this hook catches it on the way to the LLM.

More detail:

- [Design](docs/design.md): the request path, the server, and the limitations
- [Detectors](docs/detectors.md): each server mode, the native rules engine, and the token limits
- [Evaluation](docs/evaluation.md): the test suites, the canary suite, and the comparison corpus

## What gets installed

Everything goes in `~/.claude/hooks/pii/`:

- `check.sh` — the hook script, called on prompt submit and tool response
- `hook` — the native hook command both agents run, on macOS arm64 and Linux x86_64 only. It hands every case it does not settle to `check.sh`. On another platform, or when the release build cannot hand off to this `check.sh`, both agents run `check.sh`. Codex binds hook trust to the command string, so a move between the two needs trust again, and the installer warns when it happens
- `server.py` — local HTTP server that loads the selected model and returns labeled spans
- `answer.py`: builds the server's answer to a hook call, from the text to scan to the hook's exact output
- `detectors/` — one file per model backend: `redact.py`, `redact_torch.py`, `privacy_filter.py`, and `tagger.py`. See [Tagger mode](docs/detectors.md#tagger-mode)
- `rules/` — the deterministic rules: `engine.py`, `secrets.py`, `tagger_rules.py`, and `pii_rules_native.abi3.so`, the native rules engine on macOS arm64 and Linux x86_64 only. See [Native rules engine](docs/detectors.md#native-rules-engine)

An install from before this folder put the same files directly in `~/.claude/hooks/` under `pii_*` names. The installer moves its hook entries to the new paths and removes those files. It keeps `pii-check.sh` and `pii-hook` there as two-line scripts that run the current hook command, because an agent session that started before the upgrade still calls them. Codex reads its hooks once per session.

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

The installer first warns that a piped script runs as you and asks `[Y/n]` on the terminal. An agent or CI job has no terminal, so it passes `-s -- -y`.

Flags:

```bash
... | bash -s -- --prompt-only   # skip both tool hooks, keeping the prompt hook alone
... | bash -s -- --no-codex      # ignore Codex even if ~/.codex/ exists
... | bash -s -- --no-pilot      # skip the pilot warm-up run
```

The installer stops a server already running on the port, because a running server keeps the code it started with. Then, before wiring, it does a pilot run: it starts the selected server once, smoke-tests it, and leaves it warm. The OpenAI model downloads to `~/.cache/opf/`. The Redact cache must already exist. Pass `--no-pilot` to skip the pilot. The next hook call then starts the server.

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
| `standard` (default) | `secret`, `account_number`, `private_email`, `private_phone`, `private_address`, `private_username` |
| `strict` | `secret`, `account_number`, `private_email`, `private_phone`, `private_address`, `private_username`, `private_person`, `private_url`, `private_date` |

`PII_ALLOW_LABELS` accepts these label names:

| Label | Tier | Typical data |
|---|---|---|
| `secret` | critical | API keys, tokens, passwords, private keys |
| `account_number` | critical | bank accounts, card numbers, account IDs |
| `private_email` | moderate | personal or private email addresses |
| `private_phone` | moderate | phone numbers |
| `private_address` | moderate | street addresses |
| `private_username` | moderate | a person's handle or login; only tagger mode reports it |
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

## Configuration knobs

All env vars override defaults; set them in your shell or the hook's env:

| Var | Default | Purpose |
|---|---|---|
| `PII_LEVEL` | `standard` | tier (off/relaxed/standard/strict); `PII_BLOCK_LEVEL` is the legacy name |
| `PII_ALLOW_LABELS` | empty | comma-separated labels to allow within the selected tier |
| `PII_ALLOW_BYPASS` | `1` | `1` enables the `pii:off` prompt prefix; `0` disables it, for deployments where more than one person can reach the agent |
| `PII_ACTION_MODE` | `warn` | `warn` to allow input with agent context or `block` to reject input |
| `PII_SERVER_MODE` | `redact` | `redact` (LiteRT graph), `redact-torch` (checkpoint), `openai`, `rules`, or `tagger` |
| `PII_PORT` | `9123` | local server port |
| `PII_SERVER_SCRIPT` | `~/.claude/hooks/pii/server.py` | server script path |
| `PII_SERVER_LOG` | `~/.cache/pii/server.log` | server log path |
| `PII_MAX_BODY_BYTES` | `2097152` | largest request body either server reads (2 MiB); a larger declared length is answered with HTTP 413 before the body is read. A client still streaming when the answer is sent can see a broken pipe instead of the 413 |
| `OPF_CACHE_DIR` | `~/.cache/opf` | OpenAI Privacy Filter assets, this backend only |
| `OPF_MAX_TOKENS` | `1024` | OpenAI Privacy Filter request limit; larger requests fail with HTTP 413 |
| `REDACT_CACHE_DIR` | `~/.cache/redact` | converted Redact assets cache |
| `REDACT_DEVICE` | `auto` | `redact-torch` only: `cuda`, `mps`, or explicit `cpu`. `redact` is always CPU |
| `REDACT_MAX_TOKENS` | `4096` | maximum tokens in one Redact chunk |
| `REDACT_CHUNK_OVERLAP_TOKENS` | `128` | token overlap between adjacent Redact chunks |
| `REDACT_MAX_INPUT_TOKENS` | `32768` | whole-request token cap; larger requests fail with HTTP 413 before inference |
| `PII_TAGGER_DIR` | empty | `tagger` only: a local tagger folder to use instead of the release download |
| `PII_TAGGER_FP32` | empty | `tagger` only: `1` loads the fp32 graph instead of the int8 one |

## Uninstall

1. Delete the `~/.claude/hooks/pii/` folder, and `~/.claude/hooks/pii-check.sh` and `~/.claude/hooks/pii-hook` if an upgrade left them.
2. Edit `~/.claude/settings.json` and `~/.codex/hooks.json` and remove the entries that run it.

Runtime files live under `~/.cache/pii/` (the server log and the skip events).
Each backend keeps its model assets in its own cache: `~/.cache/opf/` for the
OpenAI Privacy Filter, `~/.cache/redact/` for Redact. Remove those too if you
want them gone.

An install made before the suite had its own directory kept `server.log` and
`pii-skips.log` under `~/.cache/opf/`. Move them to `~/.cache/pii/` by hand if
you want to keep the history; a stale one left behind is simply not read.

## License

None.
