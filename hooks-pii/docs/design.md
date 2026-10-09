# hooks-pii design

## Architecture

```
prompt ──> UserPromptSubmit ──> check.sh --mode prompt ──> server.py
                                       │
                                       └── blocks or warns based on PII_ACTION_MODE
                                       └── prompt sent to Anthropic if clean

Claude tool call ──> PreToolUse ──> check.sh --mode claude-pretool ──> server.py
Claude tool runs ──> PostToolUse ──> check.sh --mode claude-posttool ──> server.py
                                       │
                                       └── blocks or warns based on PII_ACTION_MODE
                                       └── response fed to LLM next turn if clean

Codex tool runs ──> PostToolUse ──> check.sh --mode codex-posttool ──> server.py
                                        │
                                        └── blocks or warns based on PII_ACTION_MODE
```

The server is auto-started on first hook call via `uv run`, then stays warm. Health check at `http://127.0.0.1:9123/health`.

A warm call is one request. The hook sends the raw hook payload to `POST /hook`, with its settings in `X-Pii-*` headers, and the server extracts the text, runs the detector, and returns the hook's stdout and stderr. bash sends it through its own `/dev/tcp` socket, so a warm call starts no other process. Where the native `hook` is installed, both agents run it in place of the script, which also saves the start of bash. `jq` and `curl` run only when no server answers, to settle locally whether there is anything to scan before the hook starts one.

For a hard boundary at the file level, see [`agent-seatbelt`](../../README.md) (the sandbox in the parent dir).

## Limitations

This is a high-recall filter, not a hard gate. Three failure modes worth knowing:

1. **Adversarial reformatting bypass.** A motivated agent (or a model trying to satisfy a "show me the file" prompt) can reshape tool output past the NER's pattern matching: `od -c file`, `base64 file`, `xxd file`, `tr a-z A-Z < file`, splitting bytes across lines, etc. Observed empirically — given a blocked `cat secrets.txt`, a model adapted within one turn to `od -c` and the byte-spread output flowed through unblocked. The NER labels patterns it recognizes; spread-out or re-encoded versions of the same content are not labeled. Content-based filtering can't close this gap without semantic execution; treat the hook as defense-in-depth alongside the file-level sandbox, not a perimeter.

2. **Codex trust requirement.** Codex CLI gates external hooks behind a per-hook trust list. Until you trust each command, Codex registers the hook in `~/.codex/hooks.json` but does not invoke it. Trust lives in `~/.codex/config.toml` under `[hooks.state]`, keyed by `<hooks.json path>:<event>:<group>:<index>`, with `enabled = true` and a `trusted_hash` for the command. Review and trust via `/hooks` in the Codex TUI. Re-running the installer after a command change requires trust again, and the installer prints a warning when it changes the Codex command, for example from `check.sh` to `hook`. A fresh live test with `codex-cli 0.154.0` confirmed both `UserPromptSubmit` and `PostToolUse` blocking for unified shell output. Warning mode also displayed the masked `systemMessage` and continued the turn. Claude Code runs both hooks without a trust step.

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

## Planned mask mode


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
