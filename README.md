# agent-seatbelt

macOS sandbox wrapper for AI coding agents. Two files, no dependencies.

Wraps `sandbox-exec` (Apple's Seatbelt) around your agent so it can't read your secrets or write outside your project, even if you run it with `--dangerously-skip-permissions`.

## Background

I ran Claude Code in unrestricted mode for months. One day it couldn't read an env var, failed a few times, then decided to read `~/.zshrc` to debug. My `.zshrc` had hardcoded API keys. They got sent to Anthropic's servers and logged in local conversation history.

Claude apologized, told me to revoke everything, and suggested better practices. That's nice. But "the AI felt bad about it" isn't a security boundary.

Built-in sandboxes (Claude's `/sandbox`, Codex's approval policies) only gate their own tools. An agent that shells out via Bash or Python bypasses all of it. OS-level enforcement can't be bypassed. The kernel doesn't care what the agent thinks it's allowed to do.

## Install

```bash
ln -sf /path/to/agent-seatbelt/sb ~/.local/bin/sb
ln -sf /path/to/agent-seatbelt/my.sb ~/my.sb
```

Symlinks so edits to the repo are immediately live. No re-copying after updates.

## Usage

```bash
cd ~/my-project
sb claude --dangerously-skip-permissions
sb npm install
sb bash

# strip secrets from environment variables too
sb -c claude --dangerously-skip-permissions
```

`-c` / `--clean-env` wipes the environment and passes through only `HOME`, `PATH`, `SHELL`, `TERM`, `LANG`, `USER`, `TMPDIR`, `SSH_AUTH_SOCK`, editor prefs, `XDG_*` dirs, and `*_BASE_URL` vars. Off by default. Turn it on if you have API keys in your shell env.

The wrapper auto-detects your git root, so running from `~/project/src/lib/` still allowlists all of `~/project/`. It refuses to run from `$HOME` directly (that would grant writes to everything).

## What it blocks

**Reads denied:** SSH private keys (public keys and config are fine), `~/.gnupg/`, `~/.aws/` (except `~/.aws/config`), `~/.docker/`, `~/.kube/`, `~/.config/gh/hosts.yml`, `~/.pypirc`, `~/.npmrc`, `~/.cargo/credentials.toml`, `~/.netrc`, `~/.env`, shell history files, Safari data, Chrome and Zen browser profiles.

**Writes denied in HOME:** `~/.ssh/`, `~/.gnupg/`, `~/.gitconfig`, and every shell init file (`~/.zshrc`, `~/.bashrc`, `~/.profile`, `~/.zprofile`, `~/.zshenv`, `~/.zlogout`, `~/.bash_profile`). An agent can't plant persistence in your shell config.

**Writes denied in project:** `.git/hooks/`, `.git/config`, `.mcp.json`, `.vscode/`, `.idea/`, and dotfile configs (`.bashrc`, `.zshrc`, `.gitconfig`).

**Writes allowed:** your project dir (minus the above), `~/.claude/`, `~/.cache/uv/`, `~/.ollama/`, `~/.Trash/`, and system temp dirs. That's it.

Network is fully open. Process execution, IPC, Mach are all allowed. This isn't trying to be a container. It's a file-level policy.

## Layering with built-in sandboxes

This works as an outer layer around any agent's own sandbox:

```
 agent-seatbelt (OS-level, file policy)
 └── agent's built-in sandbox (tool-level, network filtering)
     └── your agent process
```

The outer layer handles what the agent shouldn't touch. The inner layer handles tool-specific permissions and (optionally) network. They don't conflict. Seatbelt rules compose by intersection.

## vs agent-safehouse

[agent-safehouse](https://github.com/eugene1g/agent-safehouse) is a bigger project solving the same problem. ~50 files, a build system, auto-detection for different agents, per-toolchain profiles, per-project config files. It's well done.

Where this project is stricter:
- Granular read-denies on secrets (AWS creds with config exception, gh hosts vs settings, SSH private vs public keys, browser profiles, shell history, `.netrc`, `.npmrc`, `.cargo/credentials.toml`). agent-safehouse doesn't cover most of these.
- Write-denies inside the project (`.git/hooks`, `.git/config`, `.mcp.json`, IDE dirs). agent-safehouse grants full write to the workdir.
- Write-denies on HOME shell init files and `~/.gitconfig`. agent-safehouse relies on not granting HOME writes in the first place, but toolchain profiles can open gaps.

Where agent-safehouse does more:
- Always-on env sanitization (this project makes it opt-in).
- Per-toolchain cache dir allowlists (Node, Python, Rust, Go, etc.).
- Agent auto-detection (Claude, Codex, Amp get different profiles).
- Docker socket blocking by default.
- Optional modules for clipboard, SSH, 1Password, headless browsers.

Pick based on what you want. If you want something you can read in 10 minutes and tune to your exact setup, this is it. If you want broad coverage maintained by others, use agent-safehouse. They can also layer together.

## Customization

`my.sb` is standard [SBPL](https://reverse.put.as/wp-content/uploads/2011/09/Apple-Sandbox-Guide-v1.0.pdf). The wrapper injects three params: `_HOME`, `_PROJECT_DIR`, `_TMPDIR`. Edit the file to match your setup. Add cache dirs your tools need, block paths specific to your machine.

## Roadmap

- **Auth proxy.** A reverse proxy running outside the sandbox that injects API keys into outbound requests. The sandboxed agent only talks to `localhost`, never sees the real keys. [mitmproxy](https://mitmproxy.org/) with header injection can do this in one line. Most SDKs (OpenAI, Anthropic, etc.) already support custom `base_url`, and the `*_BASE_URL` passthrough in `--clean-env` is there for this.
- **Per-service routing.** Proxy routes by path prefix: `/openai/*` to `api.openai.com`, `/anthropic/*` to `api.anthropic.com`, etc. Each with its own key.
- **Default to clean env.** Once the proxy handles auth, `-c` becomes safe to flip on by default. API keys no longer need to be in the environment at all.

## Caveats

- macOS only. For Linux, look at [bubblewrap](https://github.com/containers/bubblewrap).
- `sandbox-exec` is technically deprecated by Apple. Still works on Sequoia, no replacement exists for third-party use.
- Network is wide open. If a secret enters the process (env var without `-c`, or fetched via credential helper), it can be exfiltrated. The sandbox operates at file paths, not content.
- Keychain access is allowed by design so credential helpers (git, AWS) work without the agent seeing raw tokens. But the agent can still perform authenticated actions like `git push`.
