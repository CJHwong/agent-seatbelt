"""Final holdout cases for an untouched generalization check."""

from __future__ import annotations

from typing import Iterable


def _make_case(
    case_number: int,
    class_name: str,
    group_name: str,
    expected_labels: Iterable[str],
    target_text: str | None,
    text: str,
    reason: str,
) -> dict[str, object]:
    return {
        "id": f"final-{case_number:03d}",
        "class": class_name,
        "group": group_name,
        "expected": list(expected_labels),
        "target": target_text,
        "text": text,
        "reason": reason,
    }


SECRET_ROWS = [
    (
        "aws_session",
        "FwoGZXIvYXdzEJ9xP4mL7kD2sF5hJ8wC1",
        "session-token FwoGZXIvYXdzEJ9xP4mL7kD2sF5hJ8wC1",
        "AWS session token in prose",
    ),
    (
        "google_oauth",
        "GOCSPX-9xP4mL7kD2sF5hJ8wC1yU6",
        "oauth client credential GOCSPX-9xP4mL7kD2sF5hJ8wC1yU6",
        "Google OAuth client secret",
    ),
    (
        "azure_signature",
        "9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "https://blob.example.test/a?sv=2026-01-01&sig=9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Signed storage URL",
    ),
    (
        "openai_key",
        "sk-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "credential sk-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "OpenAI key without field context",
    ),
    (
        "anthropic_key",
        "sk-ant-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "credential sk-ant-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Anthropic key without field context",
    ),
    (
        "stripe_webhook",
        "whsec_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "webhook secret whsec_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Stripe webhook secret",
    ),
    (
        "grafana_key",
        "glc_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Grafana API key: glc_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Grafana key with spaced field name",
    ),
    (
        "snyk_key",
        "snyk_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Snyk credential snyk_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Snyk token without assignment context",
    ),
    (
        "pulumi_key",
        "pul-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Pulumi token pul-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Pulumi token without assignment context",
    ),
    (
        "sid_cookie",
        "SessionSid9xP4mL7kD2sF5hJ8wC1",
        "Set-Cookie: sid=SessionSid9xP4mL7kD2sF5hJ8wC1; Path=/",
        "Session cookie with SID field",
    ),
    (
        "private_token",
        "PrivateToken9xP4mL7kD2sF5hJ8wC1",
        "private_token: PrivateToken9xP4mL7kD2sF5hJ8wC1",
        "Private token field",
    ),
    (
        "nested_secret",
        "Nested9xP4mL7kD2sF5hJ8wC1",
        'secret = { value = "Nested9xP4mL7kD2sF5hJ8wC1" }',
        "Structured secret value",
    ),
    (
        "spaced_api_key",
        "SpacedApi9xP4mL7kD2sF5hJ8wC1",
        "API key: SpacedApi9xP4mL7kD2sF5hJ8wC1",
        "API key with a spaced field name",
    ),
    (
        "notion_secret",
        "secret_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Notion integration secret_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Notion integration secret",
    ),
    (
        "mailgun_key",
        "key-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Mailgun API key-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Mailgun key in prose",
    ),
    (
        "github_token",
        "github_pat_11XxP4mL7kD2sF5hJ8wC1yU6rT3",
        "github_pat_11XxP4mL7kD2sF5hJ8wC1yU6rT3",
        "GitHub token without context",
    ),
    (
        "unknown_prefix",
        "tok_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "tok_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Unknown token prefix without context",
    ),
    (
        "opaque_credential",
        "Opaque9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "credential blob Opaque9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Opaque credential without a known prefix",
    ),
    (
        "authorization_value",
        "Authorization9xP4mL7kD2sF5hJ8wC1",
        '"authorization":"Authorization9xP4mL7kD2sF5hJ8wC1"',
        "Authorization field without bearer syntax",
    ),
    (
        "oauth_secret",
        "OAuthSecret9xP4mL7kD2sF5hJ8wC1",
        "oauth_client_secret=OAuthSecret9xP4mL7kD2sF5hJ8wC1",
        "OAuth client secret field",
    ),
]


CLEAN_ROWS = [
    (
        "request_id",
        None,
        "request_id=secret_9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Secret-like request ID",
    ),
    (
        "cache_id",
        None,
        "cache_id=key-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Secret-like cache ID",
    ),
    ("model_id", None, "model_id=hf_9xP4mL7kD2sF5hJ8wC1yU6rT3", "Secret-like model ID"),
    (
        "build_id",
        None,
        "build_id=EAAA9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "Secret-like build ID",
    ),
    (
        "short_provider",
        None,
        "The prefix is sk-ant but no credential follows.",
        "Incomplete provider prefix",
    ),
    (
        "placeholder",
        None,
        "secret=YOUR_SECRET; token=null; password=<PASSWORD>",
        "Placeholder values",
    ),
    ("public_cookie", None, "Set-Cookie: theme=dark; Path=/", "Public cookie value"),
    (
        "field_name",
        None,
        "The schema has an authorization field.",
        "Field name without a value",
    ),
    (
        "short_signature",
        None,
        "https://example.test/a?sig=short",
        "Short URL signature",
    ),
    ("null_csv", None, "service,token\nworker,null", "Null CSV token"),
]


AMBIGUOUS_ROWS = [
    (
        "provider_docs",
        None,
        "Docs show whsec_test_9xP4mL7kD2 for setup.",
        "Webhook documentation value",
    ),
    (
        "password_example",
        None,
        "Tutorial password=green-horse-paper-lantern.",
        "Tutorial password",
    ),
    (
        "bearer_example",
        None,
        "Example header: Bearer SAMPLE-TOKEN-ONLY-123456.",
        "Bearer documentation value",
    ),
    (
        "secret_identifier",
        None,
        "The resource is secret_9xP4mL7kD2sF5hJ8wC1yU6rT3.",
        "Secret-like resource ID",
    ),
    (
        "twilio_sid",
        None,
        "Twilio account SID AC9xP4mL7kD2sF5hJ8wC1yU6rT3.",
        "Twilio account identifier",
    ),
    (
        "key_identifier",
        None,
        "service_account.private_key_id=KeyId9xP4mL7kD2sF5hJ8wC1",
        "Private key identifier",
    ),
    (
        "signed_example",
        None,
        "https://example.test/a?sv=1&sig=PublicSignature9xP4mL7kD2sF5hJ8wC1",
        "Signed URL example",
    ),
    (
        "model_example",
        None,
        "The sample model is hf_9xP4mL7kD2sF5hJ8wC1.",
        "Model identifier example",
    ),
]


ROBUSTNESS_ROWS = [
    (
        "zero_width",
        "Secret9xP4mL7kD2sF5hJ8wC1",
        "client_secret: Secret9xP4mL7kD2sF5\u200bhJ8wC1",
        "Zero-width character in a value",
    ),
    (
        "escaped_json",
        "Secret9xP4mL7kD2sF5hJ8wC1",
        r'{"secret":"Secret\u0039xP4mL7kD2sF5hJ8wC1"}',
        "Unicode escape in JSON",
    ),
    (
        "percent_encoded",
        "Secret9xP4mL7kD2sF5hJ8wC1",
        "access_token=Secret%39xP%34mL%37kD%32sF%35hJ%38wC%31",
        "Percent-encoded token",
    ),
    (
        "split_key",
        "sk-9xP4mL7kD2sF5hJ8wC1yU6rT3",
        "const key = 'sk-9xP4mL7kD2sF5' + 'hJ8wC1yU6rT3';",
        "Split provider key",
    ),
    (
        "base64_value",
        "EncodedSecret9xP4mL7kD2sF5",
        "base64: RW5jb2RlZFNlY3JldDl4UDRtTDdrRDJzRjVoSjg=",
        "Base64 encoded value",
    ),
    (
        "html_value",
        "HtmlSecret9xP4mL7kD2sF5",
        "&lt;password&gt;HtmlSecret9xP4mL7kD2sF5&lt;/password&gt;",
        "HTML escaped field",
    ),
    (
        "spaced_key",
        "github_pat_11XxP4mL7kD2sF5hJ8wC1yU6rT3",
        "g i t h u b _ p a t _ 1 1 X x P 4 m L 7 k D 2 s F 5 h J 8 w C 1 y U 6 r T 3",
        "Provider key separated by spaces",
    ),
    (
        "reversed_value",
        "Reverse9xP4mL7kD2sF5hJ8wC1",
        "reversed_password=1Cw8Jh5Fs2Dk7Lm4Px9esreveR",
        "Reversed password",
    ),
]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_number = 1
    for group_name, target_text, text, reason in SECRET_ROWS:
        cases.append(
            _make_case(
                case_number,
                "positive",
                group_name,
                ["secret"],
                target_text,
                text,
                reason,
            )
        )
        case_number += 1
    for group_name, target_text, text, reason in CLEAN_ROWS:
        cases.append(
            _make_case(case_number, "clean", group_name, [], target_text, text, reason)
        )
        case_number += 1
    for group_name, target_text, text, reason in AMBIGUOUS_ROWS:
        cases.append(
            _make_case(
                case_number, "ambiguous", group_name, [], target_text, text, reason
            )
        )
        case_number += 1
    for group_name, target_text, text, reason in ROBUSTNESS_ROWS:
        cases.append(
            _make_case(
                case_number,
                "robustness",
                group_name,
                ["secret"],
                target_text,
                text,
                reason,
            )
        )
        case_number += 1
    return cases


FINAL_HOLDOUT_CASES = build_cases()


def validate_final_holdout_cases() -> None:
    expected_counts = {
        "positive": 20,
        "clean": 10,
        "ambiguous": 8,
        "robustness": 8,
    }
    actual_counts = {
        class_name: sum(case["class"] == class_name for case in FINAL_HOLDOUT_CASES)
        for class_name in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(f"final holdout counts differ: {actual_counts}")
    if len(FINAL_HOLDOUT_CASES) != 46:
        raise ValueError(f"final holdout has {len(FINAL_HOLDOUT_CASES)} cases")


validate_final_holdout_cases()
