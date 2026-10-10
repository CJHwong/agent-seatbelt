# hooks-pii detectors

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
PII_SERVER_MODE=redact uv run hooks-pii/server/server.py --port 9123
```

Select the checkpoint backend, which needs `~/.cache/redact/redact.pt` and an
accelerator. It selects NVIDIA `cuda` first, then Apple Metal Performance Shaders
(`mps`), and fails if no accelerator exists:

```bash
PII_SERVER_MODE=redact-torch REDACT_DEVICE=mps uv run hooks-pii/server/server.py --port 9123
```

Check the selected device:

```bash
curl -sS http://127.0.0.1:9123/health
```

The health response identifies the active mode and device:

```json
{"status":"ok","mode":"redact","device":"cpu","busy":false,"version":"<sha256>"}
```

`version` is a SHA-256 hash of every `.py` file in the server's folder and below, taken when the server starts. A running server keeps the code it started with, so a copy of new hook files, from a sync or a manual copy, leaves it on the old code. When a request fails and the version differs from a hash of the installed files, the hook stops that server and starts the installed code. It stops only a pii server process on its own port: `pii/server.py`, or `pii-server.py` from an older install.

Select OpenAI as the secondary mode. Stop the existing server before changing modes on the same port:

```bash
PII_SERVER_MODE=openai uv run hooks-pii/server/server.py --mode openai --port 9123
```

Select the rules-only mode on a machine with no accelerator, or one too slow for the model:

```bash
PII_SERVER_MODE=rules uv run hooks-pii/server/server.py --mode rules --port 9123
```

Rules mode runs the deterministic checks only. It loads no checkpoint and uses no accelerator. It reports `{"status":"ok","mode":"rules","device":"cpu"}`. It finds secrets, account numbers, emails, phone numbers, URLs, IP addresses, and dates. It does not find person names or postal addresses, because those need the neural model. On the 25-case fixture it scores 21 of 25; the four misses are the two person-name and two address cases.

Rules mode needs no dependencies. `server.py` imports the standard library alone, so the hook starts it with the system `python3` instead of `uv run`. That avoids resolving the script's declared model dependencies, which include torch. The other two modes still start under `uv run`.

In `redact-torch` the neural model runs on the selected accelerator. Tokenization, deterministic checks, and span cleanup run on the CPU, and so does everything in `redact`, whose graph is CPU only. Configure the mode with `REDACT_CACHE_DIR`, `REDACT_MIN_SCORE`, `REDACT_BATCH_SIZE` (`redact-torch` only, because the LiteRT graph cannot batch), `REDACT_MAX_TOKENS`, `REDACT_CHUNK_OVERLAP_TOKENS`, and `REDACT_MAX_INPUT_TOKENS`.

### Tagger mode

`tagger` runs a 35M-parameter token tagger trained for English, Simplified Chinese and
Traditional Chinese, including developer text: tool output, logs, configuration and chat.
It finds 18 kinds of personal data, which map onto the labels above. Every identity
number becomes `account_number`, and a person's handle becomes `private_username`.

After the tagger, rules drop what it flags by form but the policy excludes:

- documentation, licence and repository URLs
- shared mailboxes and default accounts
- code identifiers read as usernames
- public figures and places, from Wikidata lists
- names next to a public-office title

Rules also add a handle found in an `author=`, `reviewer:` or `github:` field. Only the
`secret` and `account_number` findings of the deterministic rules join the result. The
other rules flag every URL and email by shape, so they would put back what the tagger's
rules dropped.

The model is [seatbelt-pii-tagger](https://huggingface.co/cjhwong/seatbelt-pii-tagger), release 2026.10.
On first use it downloads `model.int8.onnx` (36 MB), or `model.onnx` with
`PII_TAGGER_FP32=1`, plus `tokenizer.json`, `labels.json`, `persons.txt.gz`,
`places.txt.gz` and `tagger.json`, into `~/.cache/pii-tagger`. The revision is pinned to
the release commit:

```bash
uv run hooks-pii/server/server.py --mode tagger --port 9123
```

To run your own export, point `PII_TAGGER_DIR` at a folder that holds the same files.

On Apple Silicon the server also downloads `model.mlpackage` (69 MB) and runs the network
on the Neural Engine through Core ML. It reports
`{"status":"ok","mode":"tagger","device":"neural_engine"}`. The package is fp16 and pads
each window to the shortest length it holds, from 64 to 512 tokens. The Neural Engine runs only fixed shapes. Its
spans match the fp32 network's on all but 93 of 33,479 comparison rows. The int8 graph
differs on 1,963. Everywhere else, and whenever Core ML fails to load, the server runs
the int8 graph on the CPU with onnxruntime, reports `"device":"cpu"`, and logs why Core ML
failed. `PII_TAGGER_FP32=1` always runs the fp32 graph on the CPU. coremltools ships its
Core ML bindings for Python 3.13 at most, so the server script asks uv for Python 3.13 or
older.

The same requests to a server on each runtime, p50 and p95 per request (lower is better):

| Machine | Load | Neural Engine | CPU, int8 |
|---|---|---|---|
| M5 Pro | 1 client, 1,500 texts | 1.2 ms, 9.4 ms | 2.5 ms, 12.9 ms |
| M5 Pro | 8 clients, 1,500 texts | 10.2 ms, 69.7 ms | 20.8 ms, 93.7 ms |
| M1 Pro | 1 client, 1,500 texts | 2.3 ms, 15.6 ms | 3.5 ms, 23.2 ms |
| M1 Pro | 8 clients, 1,500 texts | 15.5 ms, 119.2 ms | 28.2 ms, 180.4 ms |

On the CPU:

Each window runs in its own session call, without padding. Several calls run at once,
with 3 threads each, and the pool has one call per 3 cores. A window's spans therefore
do not depend on the other windows in the text. A single batched call would make them
depend on each other, because dynamic int8 quantizes the activations of a whole call
together. The export fuses attention, layer norm and gelu into onnxruntime's own ops.
An older export without them still loads, only slower. Measured on an 18-core M5 Pro
over 3,816 benchmark texts (higher is better):

| Graph | Before: one call of 64 windows, unfused | After: parallel single windows, fused |
|---|---|---|
| int8, one text per call | 18.6k chars/s | 47.5k chars/s |
| int8, texts of about 100k chars | 20.8k chars/s | 65.5k chars/s |
| fp32, one text per call | 18.3k chars/s | 44.3k chars/s |
| fp32, texts of about 100k chars | 20.1k chars/s | 51.4k chars/s |

### Native rules engine

The rules are Python regular expressions. On a slow CPU, Python `re` takes too long on a large tool output: 105 ms on a 12 KB input on a 2015 Celeron N3050. The native engine matches the same patterns in compiled code. `rules/engine.py` passes its own pattern strings to it, so the rules still live in one file.

It makes two changes:

- PCRE2 compiles each pattern to machine code. PCRE2 supports the lookarounds and the conditional group that the rules use.
- Each pattern names the literals it cannot match without, in `RULE_KEYWORDS`. One Aho-Corasick pass over the text finds the literals that are present. A pattern whose literals are all absent does not run. The Python engine uses the same gates.

Measured through the server on 1,000 real transcript inputs per machine. The time is the server's `processing_ms`, rules mode:

| Machine | p99 input | Python `re` at p99 | Native at p99 | Inputs under 10 ms, native |
|---|---|---|---|---|
| Apple M1 Pro | 24 KB | 13 ms | 2 ms | 100.0% |
| Celeron N3050 | 16 KB | 40 ms | 11 ms | 98.9% |

Both engines returned the same spans for all 2,000 inputs. On the Celeron, the 223 provider rules cost under 1 ms at p99 over the 71 rules of v1.0.0, measured side by side on the same inputs.

The native engine returns the same spans as `re`. PCRE2 and `re` differ in three places, and each is handled:

- **Word characters.** PCRE2 counts combining marks and connector punctuation as word characters for `\w` and `\b`, and `re` does not. Each pattern is compiled twice: once as written, and once with `\w` and `\b` spelled the way `re` reads them. A text holding one of those characters uses the spelled form, which is half as fast.
- **Unicode versions.** Python 3.11 knows Unicode 14, and PCRE2 knows Unicode 16. When the versions differ, `rules/engine.py` runs each character class over every code point in both engines at start. A text holding a character they place differently goes to `re`. This takes 0.2 s on an M1, and it is skipped when the versions match.
- **Case folding.** Four characters match an ASCII letter when case is ignored (`İ`, `ı`, `ſ`, and the Kelvin sign), and an ASCII keyword search cannot see them. A text holding one goes to `re`. So does a text with a lone surrogate, which cannot be passed to Rust.

The test suite compares the two engines on every case text in the repository, and on every code point for each character class.

The installer downloads the build for the platform from the latest GitHub release. On any other platform, or when the download fails, the rules run on the Python engine. The spans are the same, but the scan is slower. The server log names the engine at start:

```
[rules] rules engine: native
[rules] rules engine: python (No module named 'rules.pii_rules_native')
```

To build it yourself, install Rust and run:

```bash
cd hooks-pii/server/rules/native && cargo build --release
cp target/release/libpii_rules_native.dylib ../pii_rules_native.abi3.so   # macOS
cp target/release/libpii_rules_native.so ../pii_rules_native.abi3.so      # Linux
```

The file is built against the Python stable ABI, so one build loads on CPython 3.9 and later. Restart the server after you replace the file.

### Provider token patterns

`rules/secrets.py` holds 271 provider token rules for the secret types on GitHub's [secret scanning list](https://docs.github.com/en/code-security/secret-scanning/introduction/supported-secret-scanning-patterns). GitHub does not publish its own patterns. Most rules come from three MIT-licensed projects:

- 71 rules from [gitleaks](https://github.com/gitleaks/gitleaks)
- 107 rules from [betterleaks](https://github.com/betterleaks/betterleaks), with ids prefixed `betterleaks/`
- 48 rules from [microsoft/security-utilities](https://github.com/microsoft/security-utilities), with ids prefixed `microsoft/`
- 45 rules written here, with ids prefixed `seatbelt/`, for types whose format a vendor page, a vendor SDK or an open scanner states. The comment above each rule names that source.

`tests/github-secret-types.tsv` lists each GitHub type and the rule that covers it. 315 of the 470 types have a rule. A type credited to a context rule, such as `GENERIC_SECRET_CONTEXT_PATTERN`, is found when it sits under the key name its vendor documents, like `refresh_token`, and not on its own. A type whose format no source states has no rule, because a keyword and a length alone flag too much ordinary text. A rule reports a match only when the secret passes the source's entropy floor and filters, and only when one of the source's keywords is in the text. Every rule also applies the global allowlist that gitleaks and betterleaks share.

25 betterleaks rules and one `seatbelt/` rule skip a secret whose length over its token count is 2.5 or more, as betterleaks does. Words and identifiers take few tokens for their length, and a random key takes many. The count uses the `cl100k_base` vocabulary betterleaks embeds. `cl100k_base.tokens.gz` holds it in a compact form that fits the repo's 500 KB file limit. `rules/engine.py` loads it on the first secret that needs it, which takes 23 ms on an M1 and 168 ms on the Celeron. Betterleaks composite rules report only next to another rule's match, such as a client id. They are left out, except where the secret carries its own prefix, like `AIK_SECRET_`. Those report on their own.

The port leaves out the gitleaks rules that pair a keyword with any string of the right length and have no entropy floor. gitleaks `adafruit-api-key`, for one, flags `adafruit_feed = "temperature-sensor-living-room-1"`.

### Token limits and long outputs

The Redact checkpoint declares 512 position embeddings. The implementation uses 256-token model windows holding 254 content tokens, advanced with a step of 190, so adjacent windows share 64 tokens. The code names those `CONTENT_WINDOW_LENGTH`, `WINDOW_OVERLAP` and `WINDOW_STEP`, and `WINDOW_STEP` is derived from the other two rather than written out. It groups those windows into chunks of up to `REDACT_MAX_TOKENS` tokens. Longer input is chunked with `REDACT_CHUNK_OVERLAP_TOKENS` overlap, which is separate from the window overlap. Deterministic rules scan the full input before model inference. A request above `REDACT_MAX_INPUT_TOKENS` returns HTTP 413 before inference, but only after the whole text has been tokenized, so the rejection costs memory in proportion to the body rather than to the cap: a 4 MB body was measured taking 2.5 s and about 640 MB of growth to reach its 413, and a 1 MB body reached it in 0.3 s without the peak moving. Note that the cap bounds tokens, not the number of forward passes. A request at the cap is subdivided into roughly 180 model windows, so it is far more work than the token count suggests.

One asymmetry is worth knowing and is left in place deliberately. A token that sits on a window seam is covered by two windows, and the aggregation takes the maximum across them and then renormalizes, which can only lower that token's score. Seams fall every 190 tokens, so an entity landing on one scores slightly below the same entity elsewhere. The bias is small and one-directional; it is recorded in the code rather than corrected.

The OpenAI Privacy Filter checkpoint declares 131,072 position embeddings. This wrapper keeps `OPF_MAX_TOKENS=1024` as its request limit, which costs 2.28 s with one intra-op thread and 0.70 s with four. The hook's POST budget is 5 seconds and must also carry the round trip and the JSON parse, so the previous 4096 default accepted work that took 13 seconds on one thread and could never have fit. Raise it with `OPF_MAX_TOKENS` on a machine that can afford it: 4096 tokens takes about 3.9 s on four threads.

OpenAI mode does not chunk, so an oversized request returns HTTP 413 before any inference. The hook then names the size as the cause rather than blaming the detector, and tells the agent to raise the limit. In warn mode the content is allowed through unscanned, which is stated in the message; in block mode it is blocked, because a value that cannot be scanned cannot be cleared.

Review the [Redact release](https://huggingface.co/desert-ant-labs/redact/resolve/v0.4.0/README.md) and its [source-available license](https://license.desertant.com/1.0) before distribution. The release publishes `redact.tflite` and a compiled Core ML model. It does not publish `redact.pt`: every published revision is missing it, and the one commit that still lists it answers 403 from the storage layer. So `redact-torch` needs a checkpoint that was prepared separately, which is why `redact` is the default.

Measure resource use on the final holdout corpus:

```bash
uv run hooks-pii/evals/run-resource-comparison.py --mode local
uv run hooks-pii/evals/run-resource-comparison.py --mode openai
```

The report includes process CPU time, wall latency, peak resident memory, and accelerator allocation. PyTorch MPS does not expose a reliable GPU utilization percentage.
