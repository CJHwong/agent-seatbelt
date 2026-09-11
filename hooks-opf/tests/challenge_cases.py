"""Independent challenge cases for layered privacy-filter evaluation."""

from __future__ import annotations

from typing import Iterable


CaseRow = tuple[str, str | None, str, str]


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
        "id": f"challenge-{case_number:03d}",
        "class": class_name,
        "group": group_name,
        "expected": list(expected_labels),
        "target": target_text,
        "text": text,
        "reason": reason,
    }


SECRET_ROWS: list[CaseRow] = [
    (
        "cloud_credential",
        "AKIAQ7M4N8P2R6T9V3W5",
        "AWS_ACCESS_KEY_ID=AKIAQ7M4N8P2R6T9V3W5",
        "AWS access key in an environment file",
    ),
    (
        "cloud_credential",
        "mL7qV2xP9rT4nK8sD5fH1jC6wB3zY0uA",
        "aws_secret_access_key = mL7qV2xP9rT4nK8sD5fH1jC6wB3zY0uA",
        "AWS secret in an assignment",
    ),
    (
        "cloud_credential",
        "FwoGZXIvYXdzEJr8Qv9kL2mN5pR7sT1uV3wX6yZ0aB4cD8eF",
        "session_token: FwoGZXIvYXdzEJr8Qv9kL2mN5pR7sT1uV3wX6yZ0aB4cD8eF",
        "AWS session token in YAML",
    ),
    (
        "cloud_credential",
        "AzrQ7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        '{"tenant":"ops","client_secret":"AzrQ7vN4xP8mL2kD6sF9hJ3wC5bT1yU0"}',
        "Azure client secret in JSON",
    ),
    (
        "cloud_credential",
        "-----BEGIN PRIVATE KEY-----",
        '"private_key": "-----BEGIN PRIVATE KEY-----\\nFAKEKEYDATAQ7N8\\n-----END PRIVATE KEY-----"',
        "GCP service-account private-key block",
    ),
    (
        "cloud_credential",
        "ocid1.securitytoken.oc1..Q7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "OCI security token: ocid1.securitytoken.oc1..Q7vN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "OCI token in prose",
    ),
    (
        "cloud_credential",
        "cfp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "CLOUDFLARE_API_TOKEN=cfp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Cloudflare token in shell syntax",
    ),
    (
        "cloud_credential",
        "dop_v1_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "digitalocean_token: dop_v1_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "DigitalOcean token in YAML",
    ),
    (
        "cloud_credential",
        "hrk_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "heroku-api-key=hrk_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Heroku key in an application log",
    ),
    (
        "cloud_credential",
        "vercel_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "VERCEL_TOKEN: vercel_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Vercel token in a deployment record",
    ),
    (
        "cloud_credential",
        "nfp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "netlify auth token nfp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Netlify token in narrative text",
    ),
    (
        "cloud_credential",
        "sbp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "supabase_access_token=sbp_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Supabase token in an environment file",
    ),
    (
        "saas_key",
        "sk-proj-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "OpenAI project key: sk-proj-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "OpenAI project key with a new prefix",
    ),
    (
        "saas_key",
        "sk-ant-api03-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "ANTHROPIC_API_KEY=sk-ant-api03-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Anthropic key in shell syntax",
    ),
    (
        "saas_key",
        "rk_test_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "restricted Stripe test key rk_test_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Stripe restricted test key",
    ),
    (
        "saas_key",
        "EAAA7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "square_access_token = EAAA7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Square token in TOML-like text",
    ),
    (
        "saas_key",
        "SK7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Twilio auth token: SK7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Twilio token with a short provider prefix",
    ),
    (
        "saas_key",
        "SG.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0.9hJ3wC5bT1yU0",
        "SENDGRID_API_KEY=SG.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0.9hJ3wC5bT1yU0",
        "SendGrid key in shell syntax",
    ),
    (
        "saas_key",
        "key-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "mailgun key-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0 was exposed",
        "Mailgun key in a log sentence",
    ),
    (
        "saas_key",
        "sk.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Mapbox secret token: sk.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Mapbox secret token",
    ),
    (
        "saas_key",
        "FastlyTestToken7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "FASTLY_API_TOKEN='FastlyTestToken7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0'",
        "Fastly test token in a quoted assignment",
    ),
    (
        "saas_key",
        "dd_api_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "datadog_api_key: dd_api_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Datadog key in YAML",
    ),
    (
        "saas_key",
        "NRAK-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "New Relic license token NRAK-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "New Relic token in prose",
    ),
    (
        "saas_key",
        "hf_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "huggingface_token = hf_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Hugging Face token in config",
    ),
    (
        "source_control",
        "github_pat_11QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "github_pat_11QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "GitHub fine-grained token without context",
    ),
    (
        "source_control",
        "glpat-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "GitLab token glpat-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0 in a merge log",
        "GitLab token in a merge log",
    ),
    (
        "source_control",
        "bb_app_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "bitbucket_app_password=bb_app_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Bitbucket app password",
    ),
    (
        "source_control",
        "npm_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "//registry.npmjs.org/:_authToken=npm_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "npm registry token in a config line",
    ),
    (
        "source_control",
        "pypi-AgEIcHlwaS5vcmcCJDcQvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "pypi_token: pypi-AgEIcHlwaS5vcmcCJDcQvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "PyPI token in YAML",
    ),
    (
        "source_control",
        "dckr_pat_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "docker_password=dckr_pat_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Docker Hub token in an environment file",
    ),
    (
        "ci_credential",
        "cci_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "CIRCLE_TOKEN=cci_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "CircleCI token in a pipeline",
    ),
    (
        "ci_credential",
        "bk_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "buildkite agent token bk_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Buildkite token in prose",
    ),
    (
        "ci_credential",
        "travis_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "TRAVIS_TOKEN=travis_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Travis token in shell syntax",
    ),
    (
        "ci_credential",
        "codecov_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "codecov upload token: codecov_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Codecov token in a build log",
    ),
    (
        "ci_credential",
        "dp.st.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "DOPPLER_TOKEN=dp.st.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Doppler token in an environment file",
    ),
    (
        "ci_credential",
        "hvs.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "vault_token = hvs.7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Vault token in TOML",
    ),
    (
        "messaging_credential",
        "https://hooks.slack.com/services/T7QvN4/B8mL2/D6sF9hJ3wC5bT1yU0",
        "Slack webhook: https://hooks.slack.com/services/T7QvN4/B8mL2/D6sF9hJ3wC5bT1yU0",
        "Slack webhook URL",
    ),
    (
        "messaging_credential",
        "https://discord.com/api/webhooks/117QvN4/7mL2kD6sF9hJ3wC5bT1yU0",
        "discord webhook https://discord.com/api/webhooks/117QvN4/7mL2kD6sF9hJ3wC5bT1yU0",
        "Discord webhook URL",
    ),
    (
        "messaging_credential",
        "7123456789:AAQvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "TELEGRAM_BOT_TOKEN=7123456789:AAQvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Telegram bot token",
    ),
    (
        "messaging_credential",
        "secret_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "notion integration secret_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Notion integration secret",
    ),
    (
        "messaging_credential",
        "lin_api_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "LINEAR_API_KEY=lin_api_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Linear API key",
    ),
    (
        "messaging_credential",
        "sntrys_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "sentry auth token sntrys_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Sentry authentication token",
    ),
    (
        "messaging_credential",
        "webex_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "WEBEX_ACCESS_TOKEN: webex_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Webex token in a deployment value",
    ),
    (
        "messaging_credential",
        "intercom_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "intercom token is intercom_7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Intercom token in prose",
    ),
    (
        "credential_transport",
        "eyJraWQiOiJxN1YyIiwidHlwIjoiSldUIn0.eyJzdWIiOiJxN1YyIiwic2NwIjoiYWRtaW4ifQ.QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Authorization: Bearer eyJraWQiOiJxN1YyIiwidHlwIjoiSldUIn0.eyJzdWIiOiJxN1YyIiwic2NwIjoiYWRtaW4ifQ.QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Bearer JWT in an HTTP header",
    ),
    (
        "credential_transport",
        "Basic QWxhZGRpbjpRN3ZONHhQOG1MMmtENnNGOWhKM3dDNWJUMXlVMA==",
        "Authorization: Basic QWxhZGRpbjpRN3ZONHhQOG1MMmtENnNGOWhKM3dDNWJUMXlVMA==",
        "Basic authentication header",
    ),
    (
        "credential_transport",
        "p9J7qV2xP4mL8kD6",
        "postgresql://reporter:p9J7qV2xP4mL8kD6@db.internal.example:5432/ledger",
        "PostgreSQL password in a connection string",
    ),
    (
        "credential_transport",
        "mY4rT8vN2qK6sF1h",
        "mysql://batch:mY4rT8vN2qK6sF1h@mysql.internal.example:3306/jobs",
        "MySQL password in a connection string",
    ),
    (
        "credential_transport",
        "mongoP7vN4xL2kD6",
        "mongodb+srv://worker:mongoP7vN4xL2kD6@cluster.example/jobs",
        "MongoDB password in a connection string",
    ),
    (
        "credential_transport",
        "redisP7vN4xL2kD6",
        "redis://:redisP7vN4xL2kD6@cache.internal.example:6379/0",
        "Redis password in a connection string",
    ),
    (
        "generic_secret",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "client_secret: QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Generic client secret in YAML",
    ),
    (
        "generic_secret",
        "J7wC5bT1yU0QvN4xP8mL2kD6sF9hJ3",
        '{"apiKey":"J7wC5bT1yU0QvN4xP8mL2kD6sF9hJ3"}',
        "Generic API key in JSON",
    ),
    (
        "generic_secret",
        "RidgeCanyon!704",
        'password = "RidgeCanyon!704"',
        "Password in TOML",
    ),
    (
        "generic_secret",
        "T1yU0QvN4xP8mL2kD6sF9hJ3wC5b",
        "access_token = 'T1yU0QvN4xP8mL2kD6sF9hJ3wC5b'",
        "Access token in a configuration file",
    ),
    (
        "generic_secret",
        "C5bT1yU0QvN4xP8mL2kD6sF9hJ3w",
        'resource "secret" "api" { value = "C5bT1yU0QvN4xP8mL2kD6sF9hJ3w" }',
        "Terraform secret resource",
    ),
    (
        "generic_secret",
        "KubePass7QvN4xP8mL2kD6",
        "apiVersion: v1\nkind: Secret\nstringData:\n  password: KubePass7QvN4xP8mL2kD6",
        "Kubernetes secret value",
    ),
    (
        "generic_secret",
        "JavaPass7QvN4xP8mL2kD6",
        "spring.datasource.password=JavaPass7QvN4xP8mL2kD6",
        "Java properties password",
    ),
    (
        "generic_secret",
        "XmlPass7QvN4xP8mL2kD6",
        "<credential><password>XmlPass7QvN4xP8mL2kD6</password></credential>",
        "XML password value",
    ),
    (
        "generic_secret",
        "DiffPass7QvN4xP8mL2kD6",
        "+DATABASE_PASSWORD=DiffPass7QvN4xP8mL2kD6",
        "Secret in a source-control diff",
    ),
    (
        "generic_secret",
        "LogPass7QvN4xP8mL2kD6",
        "2027-09-12T04:11:20Z login password=LogPass7QvN4xP8mL2kD6 user=worker",
        "Secret in an application log",
    ),
    (
        "generic_secret",
        "CurlPass7QvN4xP8mL2kD6",
        "curl -H 'X-Api-Key: CurlPass7QvN4xP8mL2kD6' https://service.example.test",
        "Secret in a shell command",
    ),
    (
        "generic_secret",
        "npmrc_7QvN4xP8mL2kD6sF9hJ3",
        "_authToken=npmrc_7QvN4xP8mL2kD6sF9hJ3",
        "Registry token in npm configuration",
    ),
    (
        "generic_secret",
        "ActionsPass7QvN4xP8mL2kD6",
        "env:\n  DEPLOY_TOKEN: ActionsPass7QvN4xP8mL2kD6",
        "GitHub Actions environment value",
    ),
    (
        "generic_secret",
        "ComposePass7QvN4xP8mL2kD6",
        "services:\n  worker:\n    environment:\n      - SECRET=ComposePass7QvN4xP8mL2kD6",
        "Docker Compose secret value",
    ),
    (
        "generic_secret",
        "Refresh7QvN4xP8mL2kD6sF9",
        "settings = { refresh_token: 'Refresh7QvN4xP8mL2kD6sF9' }",
        "Refresh token in Python-like configuration",
    ),
    (
        "generic_secret",
        "PrivateKey7QvN4xP8mL2kD6",
        "const config = { privateKey: 'PrivateKey7QvN4xP8mL2kD6' };",
        "Private key field in JavaScript",
    ),
    (
        "generic_secret",
        "Cookie7QvN4xP8mL2kD6sF9",
        "Set-Cookie: session=Cookie7QvN4xP8mL2kD6sF9; HttpOnly; Secure",
        "Session cookie in an HTTP response",
    ),
    (
        "generic_secret",
        "Query7QvN4xP8mL2kD6sF9",
        "https://api.example.test/export?access_token=Query7QvN4xP8mL2kD6sF9",
        "Access token in a URL query",
    ),
    (
        "generic_secret",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        '{"kty":"EC","crv":"P-256","d":"QvN4xP8mL2kD6sF9hJ3wC5bT1yU0"}',
        "Private JWK parameter",
    ),
    (
        "private_key",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAAQ2hhbGxlbmdl\n-----END OPENSSH PRIVATE KEY-----",
        "OpenSSH private-key marker",
    ),
    (
        "private_key",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----\nFAKEECQvN4xP8mL2kD6\n-----END EC PRIVATE KEY-----",
        "EC private-key marker",
    ),
    (
        "credential_transport",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjaGFsbGVuZ2UifQ.QvN4xP8mL2kD6sF9hJ3",
        "cookie=session=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjaGFsbGVuZ2UifQ.QvN4xP8mL2kD6sF9hJ3",
        "JWT in a cookie",
    ),
    (
        "generic_secret",
        "Quoted7QvN4xP8mL2kD6sF9hJ3",
        'export DEPLOY_SECRET="Quoted7QvN4xP8mL2kD6sF9hJ3"',
        "Quoted multiline environment value",
    ),
    (
        "generic_secret",
        "Unknown7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "credential_value: Unknown7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "Unknown provider credential with a generic field",
    ),
    (
        "generic_secret",
        "Fence7QvN4xP8mL2kD6sF9hJ3",
        "```env\nSERVICE_SECRET=Fence7QvN4xP8mL2kD6sF9hJ3\n```",
        "Secret in a Markdown code fence",
    ),
    (
        "generic_secret",
        "Trace7QvN4xP8mL2kD6sF9hJ3",
        "RuntimeError: auth failed; client_secret=Trace7QvN4xP8mL2kD6sF9hJ3",
        "Secret in a stack trace",
    ),
    (
        "provider_key",
        "tok_7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "tok_7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "Unknown token prefix without context",
    ),
    (
        "generic_secret",
        "山谷密鑰7QvN4xP8mL2kD6",
        "部署密鑰：山谷密鑰7QvN4xP8mL2kD6",
        "Secret value beside Traditional Chinese context",
    ),
    (
        "generic_secret",
        "CSVPass7QvN4xP8mL2kD6",
        "service,environment,password\nworker,prod,CSVPass7QvN4xP8mL2kD6",
        "Secret in CSV output",
    ),
    (
        "cloud_credential",
        "AKIAZ5C1V7B3N9M2Q8R4",
        "[profile batch]\naws_access_key_id = AKIAZ5C1V7B3N9M2Q8R4\naws_secret_access_key = Profile7QvN4xP8mL2kD6",
        "AWS profile with two credentials",
    ),
]


PII_ROWS: list[tuple[str, str, str, str]] = [
    (
        "private_email",
        "private_email",
        "Send the invoice to mika.rios@blueharbor.example.",
        "Email with a new local part",
    ),
    (
        "private_email",
        "private_email",
        "Contact devon+legal@copperfield.example for the signed copy.",
        "Plus-address email",
    ),
    (
        "private_email",
        "private_email",
        "The analyst address is noa.kim@research.copperfield.example.",
        "Nested-domain email",
    ),
    (
        "private_email",
        "private_email",
        "Reply to luis.martin@atelier.example before Friday.",
        "Spanish-style name in an email",
    ),
    (
        "private_email",
        "private_email",
        "Use mei-ling.chen@harbor.example for the handoff.",
        "Hyphenated email local part",
    ),
    (
        "private_email",
        "private_email",
        "The contact is oconnor.tara@ledger.example.",
        "Apostrophe-free surname email",
    ),
    (
        "private_email",
        "private_email",
        "Forward the file to sari.nakamura@tokyo.example.",
        "Japanese-style email",
    ),
    (
        "private_email",
        "private_email",
        "Send it to zhang.wei@shanghai.example.cn.",
        "Chinese-style email",
    ),
    (
        "private_phone",
        "private_phone",
        "Call +886 2 2345 6789 after lunch.",
        "Taiwan landline",
    ),
    (
        "private_phone",
        "private_phone",
        "The London number is +44 20 7946 0958.",
        "UK phone number",
    ),
    (
        "private_phone",
        "private_phone",
        "Reach the Berlin contact at +49 30 901820.",
        "German phone number",
    ),
    (
        "private_phone",
        "private_phone",
        "Text +61 2 9374 4000 for the appointment.",
        "Australian phone number",
    ),
    (
        "private_phone",
        "private_phone",
        "The support contact uses (206) 555-0147.",
        "US phone number",
    ),
    (
        "private_phone",
        "private_phone",
        "Call 03-1234-5678 in the office directory.",
        "Japanese local phone",
    ),
    (
        "private_phone",
        "private_phone",
        "Vendor mobile: +33 1 42 68 53 00.",
        "French phone number",
    ),
    (
        "private_phone",
        "private_phone",
        "The contact number is +65 6123 4567.",
        "Singapore phone number",
    ),
    (
        "private_url",
        "private_url",
        "Review https://vault.copperfield.example/cases/17.",
        "Private application URL",
    ),
    (
        "private_url",
        "private_url",
        "Open http://10.42.7.19:9443/internal/export.",
        "Internal IP URL",
    ),
    (
        "private_url",
        "private_url",
        "Download https://files.blueharbor.example/clients/mei/report.",
        "Private file URL",
    ),
    (
        "private_url",
        "private_url",
        "The draft is at https://staging.atelier.example/review?id=Q7N4.",
        "Staging URL with query",
    ),
    (
        "private_url",
        "private_url",
        "Use https://localhost:9444/admin/records for the local check.",
        "Localhost URL",
    ),
    (
        "private_url",
        "private_url",
        "The private endpoint is https://[fd00::12]/records.",
        "IPv6 URL",
    ),
    (
        "private_date",
        "private_date",
        "My appointment is on 2027-11-19.",
        "ISO personal date",
    ),
    (
        "private_date",
        "private_date",
        "Reserve March 8, 2027 for the interview.",
        "Month-name personal date",
    ),
    (
        "private_date",
        "private_date",
        "Date of birth: 1987/04/23.",
        "Slash-form personal date",
    ),
    (
        "private_date",
        "private_date",
        "The renewal is scheduled for October 21 2028.",
        "Long month personal date",
    ),
    (
        "private_date",
        "private_date",
        "Keep 04-17-2029 free for the medical visit.",
        "Numeric personal date",
    ),
    (
        "private_date",
        "private_date",
        "The appointment starts on 2030-01-06 at 08:30.",
        "Date with time",
    ),
    (
        "private_person",
        "private_person",
        "Please add Priya Nanduri to the review invite.",
        "Indian name",
    ),
    (
        "private_person",
        "private_person",
        "Send the notice to Emil Hartmann.",
        "German name",
    ),
    (
        "private_person",
        "private_person",
        "Remind Hanae Fujimoto about the call.",
        "Japanese name",
    ),
    ("private_person", "private_person", "The owner is Chantal Moreau.", "French name"),
    (
        "private_person",
        "private_person",
        "Ask Omar El-Sayed to approve the request.",
        "Arabic name in Latin script",
    ),
    (
        "private_person",
        "private_person",
        "Invite Nguyễn Minh Anh to the session.",
        "Vietnamese name",
    ),
    (
        "private_person",
        "private_person",
        "The case belongs to Sofia Alvarez.",
        "Spanish name",
    ),
    (
        "private_person",
        "private_person",
        "Please notify Luka Petrović.",
        "South Slavic name",
    ),
    (
        "private_address",
        "private_address",
        "Ship the device to 91 Alder Lane, Portland, OR 97205.",
        "US address",
    ),
    (
        "private_address",
        "private_address",
        "Deliver to 16 Rue des Lilas, 75004 Paris.",
        "French address",
    ),
    (
        "private_address",
        "private_address",
        "Send it to Hauptstraße 27, 10115 Berlin.",
        "German address",
    ),
    (
        "private_address",
        "private_address",
        "The parcel goes to 3-5-7 Shibuya, Tokyo 150-0002.",
        "Japanese address",
    ),
    (
        "private_address",
        "private_address",
        "Mail the packet to 88 King Street, Sydney NSW 2000.",
        "Australian address",
    ),
    (
        "private_address",
        "private_address",
        "Use 12 Orchard Road, Singapore 238823.",
        "Singapore address",
    ),
    (
        "private_address",
        "private_address",
        "The destination is 台北市信義區松仁路 88 號 12 樓。",
        "Traditional Chinese address",
    ),
    (
        "private_address",
        "private_address",
        "Send the badge to 42 Via Roma, 00100 Roma.",
        "Italian address",
    ),
    (
        "account_number",
        "account_number",
        "Charge test order 4012 8888 8888 1881.",
        "Credit card number",
    ),
    (
        "account_number",
        "account_number",
        "The card is 5555-5555-5555-4444.",
        "Hyphenated credit card",
    ),
    (
        "account_number",
        "account_number",
        "Use 3782 822463 10005 for the payment record.",
        "American Express format",
    ),
    (
        "account_number",
        "account_number",
        "The bank account is 983746120058.",
        "Bank account in prose",
    ),
    (
        "account_number",
        "account_number",
        "IBAN: DE89370400440532013000.",
        "German IBAN",
    ),
    (
        "account_number",
        "account_number",
        "Routing number: 111000025.",
        "US routing number",
    ),
    (
        "account_number",
        "account_number",
        "Tax identifier: 219-84-7631.",
        "US tax identifier",
    ),
    (
        "account_number",
        "account_number",
        "Passport number P7N4X8M2L.",
        "Passport identifier",
    ),
    (
        "account_number",
        "account_number",
        "Driver license number D7N4X8M2L9.",
        "Driver license identifier",
    ),
    (
        "account_number",
        "account_number",
        "The device IMEI is 356938035643809.",
        "IMEI identifier",
    ),
    (
        "private_email",
        "private_email",
        "Escalate the request to rina.patel@ops.example.",
        "Operational contact email",
    ),
    (
        "private_phone",
        "private_phone",
        "Call the Seoul contact at +82 2 312 4567.",
        "Korean phone number",
    ),
    (
        "private_url",
        "private_url",
        "The case portal is https://cases.blueharbor.example/record/882.",
        "Case portal URL",
    ),
    (
        "private_date",
        "private_date",
        "My visa appointment is on 2031-06-14.",
        "Future personal date",
    ),
    (
        "private_person",
        "private_person",
        "Please ask 李明哲 to confirm the delivery.",
        "Chinese name in Han characters",
    ),
    (
        "account_number",
        "account_number",
        "The national identifier is 721-63-9048.",
        "National identifier",
    ),
]


CLEAN_ROWS: list[CaseRow] = [
    (
        "ordinary_prose",
        None,
        "The worker completed the migration and closed the ticket.",
        "Ordinary status text",
    ),
    (
        "ordinary_prose",
        None,
        "Review the parser output before the next release.",
        "Ordinary work instruction",
    ),
    (
        "ordinary_prose",
        None,
        "The secret scanner has no findings in this report.",
        "Security word without a value",
    ),
    (
        "ordinary_prose",
        None,
        "The account label describes a database table.",
        "Account word without account data",
    ),
    (
        "ordinary_prose",
        None,
        "Please update the token parser documentation.",
        "Token word without a token",
    ),
    ("near_miss", None, "AWS_ACCESS_KEY_ID=AKIA", "Truncated AWS prefix"),
    (
        "near_miss",
        None,
        "The token prefix is sk-proj but no value follows.",
        "Incomplete provider prefix",
    ),
    ("near_miss", None, "password: <PASSWORD>", "Placeholder password"),
    (
        "near_miss",
        None,
        "client_secret=${CLIENT_SECRET}",
        "Environment variable reference",
    ),
    ("near_miss", None, "api_key=YOUR_API_KEY", "Documentation placeholder"),
    ("near_miss", None, "Authorization: Bearer <token>", "Bearer placeholder"),
    ("near_miss", None, "-----BEGIN PUBLIC KEY-----", "Public key marker"),
    ("near_miss", None, "-----BEGIN PRIVATE KEY", "Incomplete private key marker"),
    ("technical_value", None, "build_id=7QvN4xP8mL2kD6sF9", "Build identifier"),
    (
        "technical_value",
        None,
        "trace_id=4b7f8a11-2f7b-4f52-9d93-8c44f6a0b1e2",
        "UUID trace identifier",
    ),
    (
        "technical_value",
        None,
        "sha256=4f2c9a7e11d0b4c8e6a1f3d9b7c5e2a0",
        "Short hash value",
    ),
    ("technical_value", None, "release=2027.11.19", "Dotted version number"),
    ("technical_value", None, "timestamp=2027-11-19T08:30:00Z", "Technical timestamp"),
    (
        "technical_value",
        None,
        "request_id=01K4Z7QVN4X8M2L6D9S3F5H7J",
        "ULID-like request ID",
    ),
    (
        "technical_value",
        None,
        "checksum=QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "Checksum value",
    ),
    (
        "technical_value",
        None,
        "tracking_number=9400111899223856928491",
        "Shipping tracking number",
    ),
    (
        "technical_value",
        None,
        "The test card is 0000 0000 0000 0000.",
        "Invalid card checksum",
    ),
    (
        "technical_value",
        None,
        "The short phone fixture is 555-0138.",
        "Incomplete phone value",
    ),
    (
        "technical_value",
        None,
        "The date parser rejects 2027-99-99.",
        "Invalid ISO date",
    ),
    (
        "technical_value",
        None,
        "The parser rejects February 30, 2027.",
        "Invalid month date",
    ),
    (
        "technical_value",
        None,
        "The service uses 192.0.2.44 in documentation.",
        "Reserved documentation IP",
    ),
    (
        "technical_value",
        None,
        "The local service listens on 10.42.0.7.",
        "Private network IP",
    ),
    (
        "technical_value",
        None,
        "The fixture contains VGhlIHF1aWNrIGJyb3duIGZveA==.",
        "Short encoded test data",
    ),
    ("technical_value", None, "The dependency version is 4.7.2.", "Semantic version"),
    (
        "public_data",
        None,
        "Contact support@product.example for product help.",
        "Public support email",
    ),
    (
        "public_data",
        None,
        "The documentation links to https://example.com/docs.",
        "Public documentation URL",
    ),
    (
        "public_data",
        None,
        "The public office is at 1 Main Street, Example City.",
        "Public business address",
    ),
    (
        "public_data",
        None,
        "The public hotline is +1 (202) 555-0100.",
        "Public test phone",
    ),
    (
        "public_data",
        None,
        "August is the eighth month in the calendar.",
        "Month without a date",
    ),
    (
        "structured_text",
        None,
        '{"status":"healthy","processing_ms":43,"spans":[]}',
        "Clean health response",
    ),
    ("structured_text", None, "GET /ready returned 204 in 37 ms.", "Clean access log"),
    (
        "structured_text",
        None,
        "SELECT state FROM tasks WHERE status = 'queued';",
        "Clean SQL example",
    ),
    (
        "structured_text",
        None,
        '<result status="healthy"><count>4</count></result>',
        "Clean XML",
    ),
    ("structured_text", None, "version: 2027-11-19\nstatus: ready", "Release metadata"),
    (
        "multilingual",
        None,
        "大阪のチームは午後に会議をします。",
        "Neutral Japanese text",
    ),
    (
        "multilingual",
        None,
        "高雄辦公室明天恢復營運。",
        "Neutral Traditional Chinese text",
    ),
    ("multilingual", None, "El servicio quedó disponible.", "Neutral Spanish text"),
    (
        "multilingual",
        None,
        "Le service reste prêt pour la revue.",
        "Neutral French text",
    ),
    (
        "multilingual",
        None,
        "Das Update wurde erfolgreich verarbeitet.",
        "Neutral German text",
    ),
    (
        "placeholder",
        None,
        "token=${TOKEN}; password=${PASSWORD}; secret=${SECRET}",
        "Template references",
    ),
    (
        "placeholder",
        None,
        "api_key: null\nclient_secret: null",
        "Null configuration values",
    ),
    ("placeholder", None, "password: changeme", "Common setup placeholder"),
    ("placeholder", None, "Authorization: Bearer REDACTED", "Already redacted header"),
    ("placeholder", None, "secret: [removed by policy]", "Policy placeholder"),
    (
        "ordinary_prose",
        None,
        "The schema defines api_key as a string field.",
        "Field name without a value",
    ),
]


AMBIGUOUS_ROWS: list[CaseRow] = [
    (
        "public_example",
        None,
        "The sample key is sk_test_7QvN4xP8mL2kD6sF9hJ3.",
        "Stripe test key",
    ),
    (
        "public_example",
        None,
        "The docs show github_pat_11EXAMPLEONLY for setup.",
        "GitHub documentation token",
    ),
    (
        "public_example",
        None,
        "Use password=hunter2 in the local tutorial.",
        "Tutorial password",
    ),
    (
        "public_example",
        None,
        "The example JWT is eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.QvN4.",
        "Short example JWT",
    ),
    (
        "public_example",
        None,
        "The test card is 4111 1111 1111 1111.",
        "Payment test card",
    ),
    (
        "public_example",
        None,
        "The fixture uses https://example.test/callback.",
        "Test callback URL",
    ),
    (
        "public_example",
        None,
        "Contact example.user@example.test for the demo.",
        "Demo email",
    ),
    (
        "public_example",
        None,
        "The sample bearer value is Bearer TEST-TOKEN-ONLY.",
        "Bearer test value",
    ),
    (
        "business_data",
        None,
        "The invoice reference is account-983746120058.",
        "Business reference resembling an account",
    ),
    (
        "business_data",
        None,
        "The report names Casey as the release owner.",
        "Employee name in a report",
    ),
    (
        "business_data",
        None,
        "The public office is at 1600 Pennsylvania Avenue NW.",
        "Public office address",
    ),
    ("business_data", None, "The meeting date is 2027-11-19.", "Business meeting date"),
    (
        "business_data",
        None,
        "The service endpoint is https://api.product.example/v1.",
        "Public API endpoint",
    ),
    (
        "business_data",
        None,
        "The support number is +1 (202) 555-0100.",
        "Public support number",
    ),
    (
        "business_data",
        None,
        "The record ID is 6ba7b810-9dad-11d1-80b4-00c04fd430c8.",
        "UUID record ID",
    ),
    (
        "business_data",
        None,
        "The shipment number is 9400111899223856928491.",
        "Tracking number",
    ),
    (
        "structured_example",
        None,
        '{"email":"docs@example.test","date":"2027-11-19"}',
        "Synthetic JSON sample",
    ),
    (
        "structured_example",
        None,
        "password=green-horse-paper-lantern",
        "Documentation password",
    ),
    ("structured_example", None, "api_key=YOUR_KEY_HERE", "Configuration placeholder"),
    ("structured_example", None, "secret_key=TEST_ONLY_VALUE", "Test secret field"),
    (
        "structured_example",
        None,
        "Authorization: Basic ZG9jdW1lbnQ6dGVzdA==",
        "Documentation auth header",
    ),
    (
        "structured_example",
        None,
        "client_secret: example-client-secret",
        "Example client secret",
    ),
    (
        "structured_example",
        None,
        "-----BEGIN RSA PRIVATE KEY----- in the fixture text",
        "Private-key marker in documentation",
    ),
    (
        "structured_example",
        None,
        "The hash is QvN4xP8mL2kD6sF9hJ3wC5bT1yU0.",
        "Hash-like value",
    ),
    (
        "multilingual",
        None,
        "京都オフィスの代表電話は +81 75 234 6789 です。",
        "Public Japanese office phone",
    ),
    (
        "multilingual",
        None,
        "Die öffentliche Adresse ist Lindenweg 8, 20095 Hamburg.",
        "Public German address",
    ),
    ("multilingual", None, "La réunion est le 8 mars 2027.", "French meeting date"),
    (
        "multilingual",
        None,
        "台中辦公室位於台中市西區。",
        "Office location without a street address",
    ),
    (
        "multilingual",
        None,
        "El contacto público es ana@example.test.",
        "Public Spanish contact",
    ),
    (
        "multilingual",
        None,
        "Das Demo-Passwort lautet demo-only-1234.",
        "German demo password",
    ),
]


ROBUSTNESS_ROWS: list[CaseRow] = [
    (
        "split_value",
        "AKIAQ7M4N8P2R6T9V3W5",
        "AWS key split across lines: AKIAQ7M4N8P2\\nR6T9V3W5",
        "Line break inside an AWS key",
    ),
    (
        "split_value",
        "sk-proj-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "const key = 'sk-proj-7QvN4xP8mL2kD6' + 'sF9hJ3wC5bT1yU0';",
        "Concatenated JavaScript string",
    ),
    (
        "split_value",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "client_secret: QvN4xP8mL2kD6sF9hJ3wC5bT1yU0".replace(
            "QvN4xP8mL2kD6", "QvN4xP8mL2kD6\u200b"
        ),
        "Zero-width character inside a secret",
    ),
    (
        "encoded_value",
        "sk-proj-7QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "base64: c2stcHJvai03UXZONHhQOG1MMmtENnNGOWhKM3dDNWJUMXlVMA==",
        "Base64 encoded secret",
    ),
    (
        "encoded_value",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "hex: 51764e347850386d4c326b4436734639684a33774335625431795530",
        "Hex encoded secret",
    ),
    (
        "reformatted_value",
        "github_pat_11QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "g i t h u b _ p a t _ 1 1 Q v N 4 x P 8 m L 2 k D 6 s F 9 h J 3 w C 5 b T 1 y U 0",
        "Characters separated by spaces",
    ),
    (
        "unicode_variant",
        "AKIAQ7M4N8P2R6T9V3W5",
        "ＡＫＩＡQ7M4N8P2R6T9V3W5",
        "Full-width prefix characters",
    ),
    (
        "escaped_value",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        '{"secret":"QvN4xP8mL2kD6sF9hJ3wC5bT1yU0"}'.replace("QvN4", "Q\\u0076N4"),
        "Unicode escape in JSON",
    ),
    (
        "shell_continuation",
        "CurlPass7QvN4xP8mL2kD6",
        "export PASSWORD=CurlPass7QvN4xP8mL2kD6\\\n",
        "Shell continuation after a secret",
    ),
    (
        "markdown_split",
        "Fence7QvN4xP8mL2kD6sF9hJ3",
        "`Fence7QvN4xP8mL2kD6` `sF9hJ3`",
        "Secret split across inline code spans",
    ),
    (
        "tool_output",
        "LogPass7QvN4xP8mL2kD6",
        "00000000  4c 6f 67 50 61 73 73 37 51 76 4e 34 78 50 38 6d",
        "Hex dump of a secret prefix",
    ),
    (
        "tool_output",
        "Unknown7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "VW5rbm93bjdRdk40eFA4bUwyazZGOWhKM3dDNWJUMXlVMA==",
        "Base64 command output",
    ),
    (
        "url_encoding",
        "Query7QvN4xP8mL2kD6sF9",
        "access_token=Query%37QvN%34xP%38mL%32kD%36sF%39",
        "Percent-encoded token",
    ),
    (
        "html_encoding",
        "XmlPass7QvN4xP8mL2kD6",
        "&lt;password&gt;XmlPass7QvN4xP8mL2kD6&lt;/password&gt;",
        "HTML-encoded password field",
    ),
    (
        "split_value",
        "redisP7vN4xL2kD6",
        "redis://:redisP7vN4x\\nL2kD6@cache.example/0",
        "Connection-string password split by a line break",
    ),
    (
        "indirection",
        "ActionsPass7QvN4xP8mL2kD6",
        "TOKEN_PART_A=ActionsPass7QvN4; TOKEN_PART_B=xP8mL2kD6; TOKEN=$TOKEN_PART_A$TOKEN_PART_B",
        "Shell variable composition",
    ),
    (
        "reversed_value",
        "p9J7qV2xP4mL8kD6",
        "reversed_password=6D8Lm4Px2Vq7J9p",
        "Reversed password value",
    ),
    (
        "image_boundary",
        "ComposePass7QvN4xP8mL2kD6",
        "[image omitted: screenshot contains ComposePass7QvN4xP8mL2kD6]",
        "Secret described outside text bytes",
    ),
    (
        "long_context",
        "tok_7QvN4xP8mL2kD6sF9hJ3wC5bT1",
        "prefix " + "normal context " * 80 + "tok_7QvN4xP8mL2kD6sF9hJ3wC5bT1 suffix",
        "Secret near the end of long text",
    ),
    (
        "multiple_layers",
        "QvN4xP8mL2kD6sF9hJ3wC5bT1yU0",
        "comment: client_secret=QvN4xP8mL2kD6sF9hJ3wC5bT1yU0 // redact before upload",
        "Secret in a source comment",
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
    for group_name, label, text, reason in PII_ROWS:
        cases.append(
            _make_case(
                case_number,
                "positive",
                group_name,
                [label],
                None,
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


CASES = build_cases()


def validate_cases() -> None:
    expected_counts = {
        "positive": 140,
        "clean": 50,
        "ambiguous": 30,
        "robustness": 20,
    }
    actual_counts = {
        class_name: sum(case["class"] == class_name for case in CASES)
        for class_name in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(f"challenge corpus counts differ: {actual_counts}")
    if len(CASES) != 240:
        raise ValueError(f"challenge corpus has {len(CASES)} cases")
    ids = [str(case["id"]) for case in CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("challenge corpus has duplicate IDs")


validate_cases()
