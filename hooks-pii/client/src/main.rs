//! The hook command, without a shell.
//!
//! pii-check.sh spends most of a warm call starting bash, about 5 ms on Linux and
//! 9 ms for a Homebrew bash, before it sends one request. This binary sends the
//! same request to POST /hook and prints the answer. It handles the answer the
//! server gives when the scan ran, and nothing else: a cold port, a failure status,
//! a timeout, or a setting the script would refuse all go to pii-check.sh, with the
//! same payload and arguments. The script then gives the answer it always gave, so
//! every failure message lives in one place.

use std::env;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{self, Command, Stdio};
use std::time::{Duration, Instant};

/// The script's own budget for one request, in pii-check.sh.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// Closes the stdout part and the stderr part of a /hook answer.
const SEPARATOR: u8 = 0x1e;
const SERVER_MODES: [&str; 5] = ["redact", "redact-torch", "openai", "rules", "tagger"];

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut payload = Vec::new();
    if let Err(error) = io::stdin().read_to_end(&mut payload) {
        eprintln!("pii-hook: could not read the hook payload: {error}");
        process::exit(1);
    }
    let Some(request) = request(&arguments, &payload) else {
        hand_off(&arguments, &payload);
    };
    let Some((stdout, stderr)) = answer(&request) else {
        hand_off(&arguments, &payload);
    };
    // A failed write here has no one to report to: both streams belong to the runtime.
    let _ = io::stdout().write_all(&stdout);
    let _ = io::stderr().write_all(&stderr);
}

/// The HTTP request pii-check.sh sends, or None for a call the script must judge.
fn request(arguments: &[String], payload: &[u8]) -> Option<Vec<u8>> {
    let mode = mode(arguments)?;
    let server_mode = setting("PII_SERVER_MODE", "redact");
    let action_mode = setting("PII_ACTION_MODE", "warn");
    let allow_bypass = setting("PII_ALLOW_BYPASS", "1");
    let settled = SERVER_MODES.contains(&server_mode.as_str())
        && matches!(action_mode.as_str(), "block" | "warn")
        && matches!(allow_bypass.as_str(), "0" | "1");
    if !settled {
        return None;
    }
    let level = setting("PII_LEVEL", &setting("PII_BLOCK_LEVEL", "standard"));
    let allow_labels = setting("PII_ALLOW_LABELS", "");
    let head = format!(
        "POST /hook HTTP/1.0\r\nContent-Type: application/json\r\nX-Pii-Mode: {mode}\r\nX-Pii-Level: {level}\r\nX-Pii-Allow-Labels: {allow_labels}\r\nX-Pii-Action-Mode: {action_mode}\r\nX-Pii-Allow-Bypass: {allow_bypass}\r\nX-Pii-Server-Mode: {server_mode}\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    let mut request = head.into_bytes();
    request.extend_from_slice(payload);
    Some(request)
}

/// The --mode value, read as the script reads it. None for a shape the script reads
/// differently, such as a trailing --mode with no value.
fn mode(arguments: &[String]) -> Option<String> {
    let mut mode = String::from("auto");
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        if argument == "--mode" {
            mode = remaining.next()?.clone();
        } else if let Some(value) = argument.strip_prefix("--mode=") {
            mode = value.to_string();
        }
    }
    Some(mode)
}

/// An environment value as bash reads ${NAME:-default}: empty counts as unset.
fn setting(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// stdout and stderr from a 200 answer, or None for anything else.
fn answer(request: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let response = exchange(request)?;
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let status_line = response[..split].split(|&byte| byte == b'\r').next()?;
    if status_line.split(|&byte| byte == b' ').nth(1)? != b"200" {
        return None;
    }
    let body = response[split + 4..].strip_suffix(&[SEPARATOR])?;
    let middle = body.iter().position(|&byte| byte == SEPARATOR)?;
    Some((body[..middle].to_vec(), body[middle + 1..].to_vec()))
}

/// The raw response, or None when no whole response arrived within the budget.
fn exchange(request: &[u8]) -> Option<Vec<u8>> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let port: u16 = setting("PII_PORT", "9123").parse().ok()?;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.write_all(request).ok()?;
    let mut response = Vec::new();
    let mut chunk = [0u8; 16384];
    loop {
        let remaining = deadline.checked_duration_since(Instant::now())?;
        stream.set_read_timeout(Some(remaining)).ok()?;
        match stream.read(&mut chunk).ok()? {
            0 => return Some(response),
            read => response.extend_from_slice(&chunk[..read]),
        }
    }
}

/// Run pii-check.sh from this binary's folder on the same input, and exit as it exits.
fn hand_off(arguments: &[String], payload: &[u8]) -> ! {
    let script = match env::current_exe() {
        Ok(path) => path.with_file_name("pii-check.sh"),
        Err(error) => fail(&format!("could not locate pii-check.sh: {error}")),
    };
    let mut child = match Command::new(&script)
        .args(arguments)
        .stdin(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => fail(&format!("could not run {}: {error}", display(&script))),
    };
    if let Some(mut stdin) = child.stdin.take() {
        // The script may exit before it reads everything, as it does for PII_LEVEL=off
        // on a cold port. Its own exit status is what counts.
        let _ = stdin.write_all(payload);
    }
    match child.wait() {
        Ok(status) => process::exit(status.code().unwrap_or(1)),
        Err(error) => fail(&format!("could not wait for {}: {error}", display(&script))),
    }
}

fn display(path: &Path) -> String {
    path.display().to_string()
}

/// A broken install: the scan cannot run, and the runtime shows stderr to the user.
fn fail(message: &str) -> ! {
    eprintln!("pii-hook: {message}");
    process::exit(1);
}
