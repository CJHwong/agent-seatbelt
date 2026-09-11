"""Fresh secret cases that were not used during rule design."""

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
        "id": f"holdout-{case_number:03d}",
        "class": class_name,
        "group": group_name,
        "expected": list(expected_labels),
        "target": target_text,
        "text": text,
        "reason": reason,
    }


SECRET_ROWS = [
    (
        "aws_temporary_key",
        "ASIA8Q2M4N6P8R0T2V4X6",
        "Temporary AWS access key: ASIA8Q2M4N6P8R0T2V4X6",
        "AWS temporary access key without an assignment name",
    ),
    (
        "google_client_secret",
        "GOCSPX-7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "OAuth client secret GOCSPX-7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "Google OAuth client secret",
    ),
    (
        "azure_sas",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "https://blob.example.test/file.txt?sv=2025-01-01&sig=QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Azure-style signed URL value",
    ),
    (
        "openai_legacy_key",
        "sk-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "OpenAI key sk-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Legacy OpenAI key format",
    ),
    (
        "anthropic_key",
        "sk-ant-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Anthropic credential sk-ant-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Anthropic key without an API version segment",
    ),
    (
        "stripe_webhook",
        "whsec_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Stripe webhook signing secret whsec_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Stripe webhook signing secret",
    ),
    (
        "grafana_key",
        "glc_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Grafana API key: glc_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Grafana cloud key",
    ),
    (
        "snyk_key",
        "snyk_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Snyk token snyk_7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Snyk token",
    ),
    (
        "pulumi_key",
        "pul-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "PULUMI_ACCESS_TOKEN=pul-7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Pulumi access token",
    ),
    (
        "bearer_opaque",
        "OpaqueBearer7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Authorization: Bearer OpaqueBearer7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Opaque bearer token",
    ),
    (
        "basic_auth",
        "QWxhZGRpbjpPcGFxdWVCcm93c2VyN3ZONHhQOG1MMg==",
        "Authorization: Basic QWxhZGRpbjpPcGFxdWVCcm93c2VyN3ZONHhQOG1MMg==",
        "Basic authentication value",
    ),
    (
        "jdbc_password",
        "JdbcPass7vN4xP8mL2kD6",
        "jdbc:postgresql://batch:JdbcPass7vN4xP8mL2kD6@db.example.test/jobs",
        "JDBC PostgreSQL password",
    ),
    (
        "session_cookie",
        "SidCookie7vN4xP8mL2kD6sF9",
        "Set-Cookie: sid=SidCookie7vN4xP8mL2kD6sF9; Secure; HttpOnly",
        "Session cookie with a short field name",
    ),
    (
        "private_token",
        "PrivateToken7vN4xP8mL2kD6",
        "private_token = PrivateToken7vN4xP8mL2kD6",
        "Private token field",
    ),
    (
        "aws_session_token",
        "FwoGZXIvYXdzEJ7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "AWS_SESSION_TOKEN=FwoGZXIvYXdzEJ7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "AWS session token in dotenv syntax",
    ),
    (
        "pypirc_password",
        "PypiPass7vN4xP8mL2kD6",
        "[pypi]\nusername = __token__\npassword = PypiPass7vN4xP8mL2kD6",
        "Package index password",
    ),
    (
        "hcl_secret",
        "HclSecret7vN4xP8mL2kD6",
        'secret = { value = "HclSecret7vN4xP8mL2kD6" }',
        "HCL secret value",
    ),
    (
        "api_header",
        "HeaderKey7vN4xP8mL2kD6sF9",
        "x-api-key: HeaderKey7vN4xP8mL2kD6sF9",
        "Lowercase API header",
    ),
    (
        "query_key",
        "QueryKey7vN4xP8mL2kD6sF9",
        "https://service.example.test/run?api_key=QueryKey7vN4xP8mL2kD6sF9",
        "API key in a query string",
    ),
    (
        "nested_json",
        "NestedSecret7vN4xP8mL2kD6",
        '{"auth":{"credentials":{"secret":"NestedSecret7vN4xP8mL2kD6"}}}',
        "Nested JSON secret field",
    ),
    (
        "yaml_token",
        "YamlToken7vN4xP8mL2kD6",
        "credentials:\n  service_token: YamlToken7vN4xP8mL2kD6",
        "Nested YAML token field",
    ),
    (
        "xml_api_key",
        "XmlApiKey7vN4xP8mL2kD6",
        "<config><api-key>XmlApiKey7vN4xP8mL2kD6</api-key></config>",
        "XML API key field",
    ),
]


CLEAN_ROWS = [
    (
        "long_request_id",
        "secret_7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "The request identifier is secret_7vN4xP8mL2kD6sF9hJ3wC5bT1.",
        "Long identifier with a secret-like prefix",
    ),
    (
        "cache_key",
        "key-7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "The cache key is key-7vN4xP8mL2kD6sF9hJ3wC5bT1.",
        "Long cache key",
    ),
    (
        "build_identifier",
        "SK7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "build_identifier=SK7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Build identifier with a provider-like prefix",
    ),
    (
        "project_identifier",
        "EAAA7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "project_id=EAAA7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "Project identifier with a provider-like prefix",
    ),
    (
        "model_identifier",
        "hf_7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "model_name=hf_7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "Model identifier with a provider-like prefix",
    ),
    (
        "placeholder_values",
        None,
        "password=YOUR_PASSWORD api_key=<API_KEY> token=null",
        "Configuration placeholders",
    ),
    (
        "redacted_header",
        None,
        "Authorization: Bearer REDACTED",
        "Redacted authorization header",
    ),
    (
        "public_secret_field",
        None,
        "The schema exposes a secret field for client configuration.",
        "Field name without a value",
    ),
    (
        "version_value",
        None,
        "release=sk-proj-7 and api_version=1.2.3",
        "Short version-like values",
    ),
    (
        "public_cookie",
        None,
        "Set-Cookie: theme=dark; Secure",
        "Non-sensitive cookie",
    ),
    (
        "public_url_signature",
        None,
        "The docs use https://example.test/file?sig=short for a sample.",
        "Short signed URL example",
    ),
    (
        "null_csv",
        None,
        "service,password\nworker,null",
        "Null CSV secret field",
    ),
]


AMBIGUOUS_ROWS = [
    (
        "twilio_account_sid",
        None,
        "Twilio account identifier AC7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Account identifier beside credential language",
    ),
    (
        "private_key_identifier",
        None,
        "service_account.private_key_id: PrivateKeyId7vN4xP8mL2kD6",
        "Private key identifier without private key material",
    ),
    (
        "provider_documentation",
        None,
        "The guide shows whsec_test_7vN4xP8mL2kD6 for webhook setup.",
        "Documentation webhook secret",
    ),
    (
        "tutorial_password",
        None,
        "For the local tutorial, use password=green-horse-paper-lantern.",
        "Tutorial password",
    ),
    (
        "basic_documentation",
        None,
        "The example header is Authorization: Basic QWxhZGRpbjp0ZXN0.",
        "Documentation basic authentication",
    ),
    (
        "long_secret_id",
        None,
        "The test fixture names the resource secret_7vN4xP8mL2kD6sF9hJ3wC5bT1.",
        "Secret-like resource identifier",
    ),
    (
        "sample_bearer",
        None,
        "The sample bearer value is Bearer SAMPLE-TOKEN-ONLY-123456.",
        "Sample bearer value",
    ),
    (
        "sample_query_key",
        None,
        "The docs use https://example.test/run?api_key=YOUR_KEY_HERE.",
        "Query parameter placeholder",
    ),
    (
        "public_csv",
        None,
        "service,password\nworker,training-password",
        "Training CSV password",
    ),
    (
        "test_model_id",
        None,
        "The test model is hf_7vN4xP8mL2kD6sF9hJ3.",
        "Model identifier in documentation",
    ),
]


ROBUSTNESS_ROWS = [
    (
        "zero_width_context",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "client_secret: QvN4xP8mL2kD6sF9\u200bhJ3wC5bT1",
        "Zero-width character inside a field value",
    ),
    (
        "unicode_escape",
        "EscapedSecret7vN4xP8mL2kD6",
        r'{"secret":"Escaped\u0053ecret7vN4xP8mL2kD6"}',
        "Unicode escape in a JSON value",
    ),
    (
        "percent_escape",
        "PercentSecret7vN4xP8mL2kD6",
        "access_token=Percent%53ecret7vN%34xP%38mL%32kD%36",
        "Percent-encoded token value",
    ),
    (
        "split_provider",
        "sk-ant-7vN4xP8mL2kD6sF9hJ3wC5bT1",
        "const key = 'sk-ant-7vN4xP8mL2kD6' + 'sF9hJ3wC5bT1';",
        "Provider key split across source strings",
    ),
    (
        "base64_context",
        "client_secret=EncodedSecret7vN4xP8mL2kD6",
        "base64: Y2xpZW50X3NlY3JldD1FbmNvZGVkU2VjcmV0N3ZONHhQOG1MMmtENg==",
        "Base64 encoded assignment",
    ),
    (
        "html_context",
        "HtmlSecret7vN4xP8mL2kD6",
        "&lt;password&gt;HtmlSecret7vN4xP8mL2kD6&lt;/password&gt;",
        "HTML escaped password field",
    ),
    (
        "reformatted_prefix",
        "github_pat_11QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "g i t h u b _ p a t _ 1 1 Q v N 4 x P 8 m L 2 k D 6 s F 9 h J 3 w C 5 b T 1",
        "Provider key separated by spaces",
    ),
    (
        "reversed_secret",
        "ReverseSecret7vN4xP8mL2kD6",
        "reversed_password=6Dk2Lm8Px4Nv7terceSesreveR",
        "Reversed password value",
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
            _make_case(
                case_number,
                "clean",
                group_name,
                [],
                target_text,
                text,
                reason,
            )
        )
        case_number += 1
    for group_name, target_text, text, reason in AMBIGUOUS_ROWS:
        cases.append(
            _make_case(
                case_number,
                "ambiguous",
                group_name,
                [],
                target_text,
                text,
                reason,
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


HOLDOUT_CASES = build_cases()


def validate_holdout_cases() -> None:
    expected_counts = {
        "positive": 22,
        "clean": 12,
        "ambiguous": 10,
        "robustness": 8,
    }
    actual_counts = {
        class_name: sum(case["class"] == class_name for case in HOLDOUT_CASES)
        for class_name in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(f"holdout corpus counts differ: {actual_counts}")
    if len(HOLDOUT_CASES) != 52:
        raise ValueError(f"holdout corpus has {len(HOLDOUT_CASES)} cases")


validate_holdout_cases()
