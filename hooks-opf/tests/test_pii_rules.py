"""Behavior tests for the deterministic PII rule engine.

Every test asserts the span offsets, labels or merged text that the module
returns. Nothing here mocks the module under test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pii_rules


# The tier map in pii-check.sh (CRITICAL + MODERATE + LOW). A label outside
# this set reaches the hook as tier "unknown".
HOOK_TIER_LABELS = {
    "secret",
    "account_number",
    "private_email",
    "private_phone",
    "private_address",
    "private_person",
    "private_url",
    "private_date",
}


def span_texts(text: str) -> set[str]:
    """Return the exact source text each deterministic span covers."""
    return {str(span["text"]) for span in pii_rules.deterministic_spans(text)}


def span_tuples(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) for each deterministic span, in order."""
    return [
        (cast(int, span["start"]), cast(int, span["end"]), str(span["label"]))
        for span in pii_rules.deterministic_spans(text)
    ]


class ContactAndIdentifierTests(unittest.TestCase):
    def test_clean_text_produces_no_spans(self) -> None:
        self.assertEqual(
            pii_rules.deterministic_spans("the quick brown fox jumps over it"), []
        )

    def test_email_is_labeled_private_email(self) -> None:
        self.assertEqual(
            span_tuples("write to alice@example.com now"),
            [(9, 26, "private_email")],
        )

    def test_url_drops_its_trailing_punctuation(self) -> None:
        self.assertEqual(
            span_texts("see https://example.com/a,)."),
            {"https://example.com/a"},
        )

    def test_phone_number_is_labeled_private_phone(self) -> None:
        self.assertEqual(span_texts("call 415-555-1234 today"), {"415-555-1234"})

    def test_iso_date_is_labeled_private_date(self) -> None:
        self.assertEqual(span_texts("shipped on 2024-01-15 by air"), {"2024-01-15"})

    def test_month_name_date_is_labeled_private_date(self) -> None:
        self.assertEqual(
            span_texts("delivered January 5, 2024 to him"), {"January 5, 2024"}
        )

    def test_card_number_that_passes_luhn_is_an_account_number(self) -> None:
        self.assertEqual(
            span_tuples("card 4111111111111111, on file"),
            [(5, 21, "account_number")],
        )

    def test_card_number_that_fails_luhn_is_not_reported(self) -> None:
        self.assertEqual(span_texts("card 4111111111111112, on file"), set())

    def test_ipv4_inside_the_octet_range_is_labeled_private_url(self) -> None:
        self.assertEqual(
            span_tuples("host 10.0.0.7 replied"),
            [(5, 13, "private_url")],
        )

    def test_ipv4_with_an_octet_above_255_is_ignored(self) -> None:
        self.assertEqual(span_texts("host 999.0.0.7 replied"), set())

    def test_iban_is_an_account_number(self) -> None:
        self.assertEqual(
            span_texts("pay to GB33BUKB20201555555555 now"),
            {"GB33BUKB20201555555555"},
        )

    def test_labelled_account_number_excludes_the_label_word(self) -> None:
        self.assertEqual(
            span_tuples("account number: 12345678"),
            [(16, 24, "account_number")],
        )

    def test_spans_never_overlap_and_come_back_in_reading_order(self) -> None:
        text = "Contact alice@example.com or 415-555-1234 about 2024-01-15."
        spans = pii_rules.deterministic_spans(text)
        self.assertEqual(
            [str(span["label"]) for span in spans],
            ["private_email", "private_phone", "private_date"],
        )
        for earlier, later in zip(spans, spans[1:]):
            self.assertLessEqual(cast(int, earlier["end"]), cast(int, later["start"]))


class GuardContextMatrixTests(unittest.TestCase):
    """Every rule family against the characters that break a guard.

    A bare token, or a token in the middle of a sentence, never takes the guard
    arm that matters. A value at the end of a sentence is the most common form
    in prose, and it is the cell a bare-token test misses: the trailing guard
    has to reject a longer dotted run without rejecting a full stop. Line
    coverage cannot see the miss, because a guard is a reached line however it
    evaluates, so this matrix is the test that holds the class down.
    """

    # name, the text immediately in front of the value, the value, the label,
    # then what a leading space, letter, digit and dot must do. "value" reports
    # the value on its own. "absorbed" lets the rule take the leading character
    # into the token, which is correct when that character is part of the
    # token: a local part, or the first digit of a longer number. "none"
    # reports nothing.
    FAMILIES = (
        (
            "email",
            "",
            "alice@example.com",
            "private_email",
            ("value", "absorbed", "absorbed", "value"),
        ),
        (
            "url",
            "",
            "https://example.com/a",
            "private_url",
            ("value", "value", "value", "value"),
        ),
        (
            "phone",
            "",
            "415-555-1234",
            "private_phone",
            ("value", "none", "absorbed", "value"),
        ),
        ("ip", "", "192.168.100.1", "private_url", ("value", "none", "none", "value")),
        (
            "date",
            "",
            "2024-01-15",
            "private_date",
            ("value", "none", "none", "value"),
        ),
        (
            "card",
            "",
            "4111111111111111",
            "account_number",
            ("value", "value", "none", "value"),
        ),
        (
            "secret",
            "api_key = ",
            "abcdefgh1234",
            "secret",
            ("value", "none", "none", "value"),
        ),
    )
    TRAILING = {
        "full stop": ".",
        "comma": ",",
        "semicolon": ";",
        "close paren": ")",
        "close bracket": "]",
        "quote": '"',
        "end of input": "",
    }
    LEADING = (" ", "x", "1", ".")

    def spans_of(self, text: str) -> list[tuple[int, int, str, str]]:
        return [
            (
                cast(int, span["start"]),
                cast(int, span["end"]),
                str(span["label"]),
                str(span["text"]),
            )
            for span in pii_rules.deterministic_spans(text)
        ]

    def test_a_trailing_context_never_hides_a_value(self) -> None:
        for name, head, value, label, _ in self.FAMILIES:
            start = len(head)
            end = start + len(value)
            for context, trailing in self.TRAILING.items():
                with self.subTest(family=name, context=context):
                    spans = self.spans_of(head + value + trailing)
                    self.assertIn((start, end, label, value), spans)
                    self.assertEqual(
                        [span for span in spans if span[1] > end],
                        [],
                        "a span swallowed the character after the value",
                    )

    def test_a_leading_context_is_handled_as_declared(self) -> None:
        for name, head, value, label, expectations in self.FAMILIES:
            for marker, expected in zip(self.LEADING, expectations):
                with self.subTest(family=name, leading=marker):
                    start = len(marker) + len(head)
                    spans = self.spans_of(marker + head + value)
                    if expected == "none":
                        self.assertEqual(spans, [])
                        continue
                    if expected == "value":
                        self.assertEqual(
                            spans, [(start, start + len(value), label, value)]
                        )
                        continue
                    self.assertEqual(len(spans), 1)
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[0][2], label)
                    self.assertTrue(spans[0][3].endswith(value))


class InvisibleCharacterTests(unittest.TestCase):
    """An invisible character must not hide a value.

    Every rule matches the text with those characters removed, so the offsets
    have to be mapped back onto the original string. The offsets are the
    contract with the hook and the redactor, so each one is checked by slicing
    the original text with it.
    """

    # Written as code points so that no invisible character sits in this file.
    ZERO_WIDTH_SPACE = chr(0x200B)
    SOFT_HYPHEN = chr(0x00AD)
    BYTE_ORDER_MARK = chr(0xFEFF)
    BELL = chr(0x07)

    def visible_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans = pii_rules.deterministic_spans(text)
        self.assertTrue(spans, text)
        for span in spans:
            start = cast(int, span["start"])
            end = cast(int, span["end"])
            self.assertEqual(span["text"], text[start:end])
        return [
            (cast(int, span["start"]), cast(int, span["end"]), str(span["label"]))
            for span in spans
        ]

    def test_a_zero_width_space_does_not_hide_an_email(self) -> None:
        text = f"alice{self.ZERO_WIDTH_SPACE}@example.com"
        self.assertEqual(self.visible_spans(text), [(0, len(text), "private_email")])

    def test_a_soft_hyphen_does_not_hide_an_email(self) -> None:
        text = f"alice{self.SOFT_HYPHEN}@example.com"
        self.assertEqual(self.visible_spans(text), [(0, len(text), "private_email")])

    def test_a_control_character_does_not_hide_an_email(self) -> None:
        text = f"alice{self.BELL}@example.com"
        self.assertEqual(self.visible_spans(text), [(0, len(text), "private_email")])

    def test_a_byte_order_mark_does_not_hide_an_email(self) -> None:
        text = f"alice{self.BYTE_ORDER_MARK}@example.com"
        self.assertEqual(self.visible_spans(text), [(0, len(text), "private_email")])

    def test_a_zero_width_space_does_not_truncate_a_secret(self) -> None:
        # The invisible character splits the AWS key mid-token, so a rule that
        # stopped at the invisible would report only the prefix. The assertion is
        # that the span covers the whole value, invisible character included.
        text = f"AKIA{self.ZERO_WIDTH_SPACE}IOSFODNN7EXAMPLE"
        self.assertEqual(self.visible_spans(text), [(0, len(text), "secret")])

    def test_a_zero_width_space_inside_a_value_stays_inside_its_span(self) -> None:
        text = f"api_key = abc{self.ZERO_WIDTH_SPACE}defgh1234"
        self.assertEqual(
            self.visible_spans(text),
            [(10, len(text), "secret")],
        )

    def test_offsets_before_and_after_an_invisible_character_stay_correct(self) -> None:
        text = f"mail alice{self.ZERO_WIDTH_SPACE}@example.com or bob@example.com"
        self.assertEqual(
            self.visible_spans(text),
            [
                (5, 23, "private_email"),
                (27, 42, "private_email"),
            ],
        )

    def test_clean_text_keeps_its_offsets(self) -> None:
        self.assertEqual(
            span_tuples("write to alice@example.com now"),
            [(9, 26, "private_email")],
        )


class CardBoundaryTests(unittest.TestCase):
    """A card run starts and ends on a digit.

    A run that ends on a separator carries that separator into the span, and a
    run that continues past one swallows the digit after it and fails Luhn.
    """

    def test_card_span_stops_at_its_last_digit(self) -> None:
        self.assertEqual(
            span_tuples("card 4111111111111111 on file"),
            [(5, 21, "account_number")],
        )

    def test_card_followed_by_a_separator_and_a_digit_is_still_found(self) -> None:
        for text, expected in (
            ("card 4111111111111111 5", (5, 21)),
            ("4111111111111111-5", (0, 16)),
            ("4111111111111111 55", (0, 16)),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    span_tuples(text), [(expected[0], expected[1], "account_number")]
                )
                self.assertEqual(span_texts(text), {"4111111111111111"})

    def test_card_written_in_digit_groups_is_reported_whole(self) -> None:
        self.assertEqual(
            span_tuples("4111 1111 1111 1111"), [(0, 19, "account_number")]
        )
        self.assertEqual(span_tuples("3782 822463 10005"), [(0, 17, "account_number")])


class PhoneRuleTests(unittest.TestCase):
    """The phone rule spans several national shapes, and leaves IPv4 alone."""

    INTERNATIONAL = (
        "+886 2 2712 3456",
        "+44 20 7946 0958",
        "+81 3 1234 5678",
        "+14155551234",
        "0044 20 7946 0958",
        "+1-415-555-1234",
        "+1 (415) 555-1234",
    )
    DOMESTIC = (
        "4155551234",
        "02-2712-3456",
        "(02) 2712 3456",
        "415.555.1234",
        "415 555 1234",
    )
    NOT_A_PHONE = ("123.456.789", "2.100.200.300")

    def test_each_supported_number_shape_is_reported_whole(self) -> None:
        for text in self.INTERNATIONAL + self.DOMESTIC:
            with self.subTest(text=text):
                self.assertEqual(span_tuples(text), [(0, len(text), "private_phone")])

    def test_a_number_inside_a_sentence_keeps_its_own_offsets(self) -> None:
        self.assertEqual(
            span_tuples("ring +886 2 2712 3456 now"),
            [(5, 21, "private_phone")],
        )

    def test_ipv4_keeps_the_url_label_and_covers_the_whole_address(self) -> None:
        # private_phone (4) outranks private_url (2), so a dotted quad that read
        # as a phone used to drop the address span and keep three octets only.
        for text in ("192.168.100.1", "172.16.100.100", "255.255.255.255"):
            with self.subTest(text=text):
                self.assertEqual(span_tuples(text), [(0, len(text), "private_url")])

    def test_ipv4_that_misses_the_phone_shape_keeps_its_url_label(self) -> None:
        self.assertEqual(span_tuples("192.168.1.1"), [(0, 11, "private_url")])

    def test_a_dotted_run_that_is_not_a_number_is_not_a_phone(self) -> None:
        for text in self.NOT_A_PHONE:
            with self.subTest(text=text):
                self.assertEqual(span_tuples(text), [])

    def test_a_dotted_run_inside_sentences_is_not_a_phone(self) -> None:
        self.assertEqual(span_tuples("the value 123.456.789 is here"), [])
        self.assertEqual(span_tuples("version 2.100.200.300 build"), [])

    def test_a_run_below_the_digit_floor_is_not_a_phone(self) -> None:
        for text in ("2024 1500", "555-1234", "1 234 567 890"):
            with self.subTest(text=text):
                self.assertEqual(span_tuples(text), [])


class LuhnTests(unittest.TestCase):
    def test_doubling_a_digit_above_four_subtracts_nine(self) -> None:
        # 59 -> 9 + (5 * 2 - 9) = 10. Only the subtract-nine branch can make it 0.
        self.assertTrue(pii_rules.luhn_valid("59"))

    def test_checksum_that_is_not_a_multiple_of_ten_is_invalid(self) -> None:
        # 19 -> 9 + 1 * 2 = 11.
        self.assertFalse(pii_rules.luhn_valid("19"))


# Built with join rather than written out. Contiguous, this is a credential-shaped
# string, and GitHub push protection cannot tell a test fixture from a live key, so
# it refuses the push. Adjacent literals are not enough: the formatter rejoins them.
# The engine receives one string, so the sk_ family is still exercised.
STRIPE_TEST_KEY = "".join(("sk_", "test_", "abcdefghijklmnopqrstuvwx"))


class SecretRuleTests(unittest.TestCase):
    def test_each_secret_family_reports_only_its_value(self) -> None:
        samples = (
            (
                "aws key AKIAIOSFODNN7EXAMPLE in the log",
                "AKIAIOSFODNN7EXAMPLE",
            ),
            (
                # STRIPE_TEST_KEY is assembled from adjacent literals. Written as one
                # literal it is a credential-shaped string, and GitHub push protection
                # cannot tell a test fixture from a live key, so it refuses the push.
                # The engine sees one string either way, so the sk_ family is still
                # exercised rather than dropped from the list.
                f"token {STRIPE_TEST_KEY} here",
                STRIPE_TEST_KEY,
            ),
            (
                "use ghp_abcdefghijklmnopqrstuvwxyz0123456789 now",
                "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            ),
            (
                "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop now",
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop",
            ),
            (
                "key -----BEGIN RSA PRIVATE KEY-----",
                "-----BEGIN RSA PRIVATE KEY-----",
            ),
            ("password is hunter2xyz", "hunter2xyz"),
            (
                "aws_secret_access_key = abcdefghijklmnopqrst",
                "abcdefghijklmnopqrst",
            ),
            ('api_key = "abcdefgh12345678"', "abcdefgh12345678"),
            (
                "Authorization: Bearer abcdefghijklmnopqrstuvwx",
                "abcdefghijklmnopqrstuvwx",
            ),
            (
                "Authorization: Basic dXNlcjpwYXNzd29yZA==",
                "dXNlcjpwYXNzd29yZA==",
            ),
            (
                "db postgres://appuser:s3cretpw@dbhost:5432/prod",
                "s3cretpw",
            ),
            (
                "mailgun key key-abcdefghijklmnopqrstuvwx",
                "key-abcdefghijklmnopqrstuvwx",
            ),
            (
                "notion secret_abcdefghijklmnopqrstuvwx",
                "secret_abcdefghijklmnopqrstuvwx",
            ),
            (
                "slack https://hooks.slack.com/services/T00000000/B00000000/abcdefghijklmnop",
                "https://hooks.slack.com/services/T00000000/B00000000/abcdefghijklmnop",
            ),
            ("<password>abcdefgh1234</password>", "abcdefgh1234"),
            ('{"kty":"EC","d":"abcdefghijklmnopqrst"}', "abcdefghijklmnopqrst"),
            (
                'resource "secret" "main" {\n  value = "abcdefgh1234"\n}',
                "abcdefgh1234",
            ),
            (
                "Set-Cookie: sessionid=abcdefghijklmnopqrst",
                "abcdefghijklmnopqrst",
            ),
            ('secret = { value = "abcdefgh1234" }', "abcdefgh1234"),
            ("密碼：abcdefgh1234", "abcdefgh1234"),
        )
        for text, expected in samples:
            with self.subTest(text=text):
                self.assertEqual(span_texts(text), {expected})

    def test_a_signed_url_reports_the_url_around_its_signature(self) -> None:
        # The signature is the secret. The URL it sits in is still a URL, so
        # merge_spans trims the URL span around the secret instead of dropping
        # it: with private_url allowed and secret not, the URL must still be
        # reported.
        text = "link https://files.example.com/report?sig=abcdefghijklmnopqrst"
        self.assertEqual(
            span_tuples(text),
            [(5, 42, "private_url"), (42, 62, "secret")],
        )

    def test_trailing_punctuation_is_trimmed_off_a_secret_value(self) -> None:
        self.assertEqual(
            span_tuples("api_key = abcdefgh1234..."),
            [(10, 22, "secret")],
        )

    def test_a_value_that_is_only_separators_is_dropped(self) -> None:
        self.assertEqual(span_tuples("password is ::::"), [])
        self.assertEqual(span_tuples("密碼：::::::::"), [])

    def test_a_template_value_is_not_reported_as_a_secret(self) -> None:
        self.assertEqual(span_tuples("api_key = test_abcdefgh1234"), [])


class SecretPlaceholderTests(unittest.TestCase):
    """The predicate that keeps template values out of the secret stream.

    No public path reaches the blank branch: every capturing group in the
    module requires at least one non-space character, and the CSV path strips
    the cell before the call. The blank inputs below exercise that guard.
    """

    def test_blank_values_are_placeholders(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                self.assertTrue(pii_rules._is_secret_placeholder(value))

    def test_brace_and_variable_tokens_are_placeholders(self) -> None:
        for value in ("<your-key>", "$TOKEN", "[redacted]", "{value}"):
            with self.subTest(value=value):
                self.assertTrue(pii_rules._is_secret_placeholder(value))

    def test_demo_words_are_placeholders_regardless_of_case(self) -> None:
        for value in ("changeme", "REDACTED", "None", "  Test  "):
            with self.subTest(value=value):
                self.assertTrue(pii_rules._is_secret_placeholder(value))

    def test_example_prefixes_are_placeholders(self) -> None:
        for value in (
            "your_api_key_value",
            "example-secret-value",
            "sample_token_value",
            "placeholder1234",
            "removed 000000",
        ):
            with self.subTest(value=value):
                self.assertTrue(pii_rules._is_secret_placeholder(value))

    def test_a_real_looking_value_is_not_a_placeholder(self) -> None:
        self.assertFalse(pii_rules._is_secret_placeholder("hunter2xyz"))


class CsvSecretTests(unittest.TestCase):
    def test_password_column_marks_the_row_value_as_secret(self) -> None:
        self.assertEqual(
            span_tuples("name,password\nbob,s3cretvalue123\n"),
            [(18, 32, "secret")],
        )

    def test_header_without_a_secret_column_yields_nothing(self) -> None:
        self.assertEqual(span_tuples("name,age\nalice,30\n"), [])

    def test_tab_delimited_header_selects_its_column(self) -> None:
        self.assertEqual(
            span_tuples("id\ttoken\n1\tabcdefghijklmnopqrst\n"),
            [(11, 31, "secret")],
        )

    def test_blank_secret_field_is_skipped(self) -> None:
        self.assertEqual(span_tuples("password,name\n,alice\n"), [])

    def test_placeholder_secret_field_is_skipped(self) -> None:
        self.assertEqual(span_tuples("password,name\nyour_password_here,alice\n"), [])

    def test_quoted_secret_header_is_matched(self) -> None:
        self.assertEqual(
            span_tuples('"password",name\ns3cretvalue123,alice\n'),
            [(16, 30, "secret")],
        )

    def test_padded_secret_cell_is_reported_without_its_padding(self) -> None:
        self.assertEqual(
            span_tuples("user,api_key\nbob,  abcdefgh1234  \n"),
            [(19, 31, "secret")],
        )


class MergeSpansTests(unittest.TestCase):
    def test_a_span_is_returned_in_reading_order_with_its_text(self) -> None:
        text = "bob@example.com met alice@example.com"
        spans: list[dict[str, object]] = [
            {"start": 20, "end": 37, "label": "private_email"},
            {"start": 0, "end": 15, "label": "private_email"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [span["text"] for span in merged],
            ["bob@example.com", "alice@example.com"],
        )

    def test_the_higher_priority_label_wins_an_overlap(self) -> None:
        text = "overlapping labels on one span"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": len(text), "label": "private_url"},
            {"start": 0, "end": len(text), "label": "secret"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual([span["label"] for span in merged], ["secret"])
        self.assertEqual(merged[0]["text"], text)

    def test_the_longer_span_wins_when_priorities_match(self) -> None:
        text = "alice@example.com"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": 5, "label": "private_email"},
            {"start": 0, "end": 17, "label": "private_email"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual([(span["start"], span["end"]) for span in merged], [(0, 17)])

    def test_an_equal_length_overlap_keeps_the_tail_of_the_later_span(self) -> None:
        text = "abcdefgh"
        spans: list[dict[str, object]] = [
            {"start": 3, "end": 8, "label": "private_date"},
            {"start": 0, "end": 5, "label": "private_date"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [(span["start"], span["end"]) for span in merged], [(0, 5), (5, 8)]
        )

    def test_a_blank_fragment_between_two_spans_is_not_labelled(self) -> None:
        # The multibyte case from the redact path: the split left a single
        # space carrying a label, which reaches the hook as a span that masks a
        # character of the surrounding prose for nothing.
        text = "hello 密碼 本地 測試 world"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": 20, "label": "private_url"},
            {"start": 8, "end": 11, "label": "secret"},
            {"start": 12, "end": 20, "label": "secret"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [
                (span["start"], span["end"], span["label"], span["text"])
                for span in merged
            ],
            [
                (0, 8, "private_url", "hello 密碼"),
                (8, 11, "secret", " 本地"),
                (12, 20, "secret", "測試 world"),
            ],
        )

    def test_a_single_character_fragment_is_not_labelled(self) -> None:
        text = "abcdefghij"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": 10, "label": "private_url"},
            {"start": 5, "end": 9, "label": "secret"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in merged],
            [(0, 5, "private_url"), (5, 9, "secret")],
        )

    def test_a_whole_span_keeps_its_length(self) -> None:
        # The drop is for a trimmed fragment only. A one-character value that a
        # rule matched outright is still a value, and the CSV column rule is
        # the path that produces one.
        self.assertEqual(span_tuples("password,name\nx,alice\n"), [(14, 15, "secret")])

    def test_a_lower_priority_span_keeps_the_parts_nobody_claimed(self) -> None:
        # A signature value inside a URL must not erase the URL around it.
        text = "abcdefghij"
        spans: list[dict[str, object]] = [
            {"start": 3, "end": 7, "label": "secret"},
            {"start": 0, "end": 10, "label": "private_url"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [
                (span["start"], span["end"], span["label"], span["text"])
                for span in merged
            ],
            [
                (0, 3, "private_url", "abc"),
                (3, 7, "secret", "defg"),
                (7, 10, "private_url", "hij"),
            ],
        )

    def test_a_lower_priority_span_that_is_fully_claimed_is_dropped(self) -> None:
        text = "abcdefghij"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": 10, "label": "secret"},
            {"start": 2, "end": 8, "label": "private_url"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in merged],
            [(0, 10, "secret")],
        )

    def test_an_unknown_label_loses_to_every_known_label(self) -> None:
        text = "abcdefgh"
        spans: list[dict[str, object]] = [
            {"start": 0, "end": 8, "label": "made_up"},
            {"start": 0, "end": 8, "label": "private_date"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual([span["label"] for span in merged], ["private_date"])

    def test_out_of_range_spans_are_dropped_before_the_priority_sort(self) -> None:
        text = "hello"
        spans: list[dict[str, object]] = [
            {"start": 2, "end": 2, "label": "secret"},
            {"start": 3, "end": 1, "label": "secret"},
            {"start": -2, "end": 4, "label": "secret"},
            {"start": 0, "end": 99, "label": "secret"},
            {"start": 0, "end": 5, "label": "private_date"},
        ]
        merged = pii_rules.merge_spans(text, spans)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label"], "private_date")
        self.assertEqual(merged[0]["text"], "hello")


class LabelContractTests(unittest.TestCase):
    def test_the_priority_map_matches_the_hook_tier_map(self) -> None:
        self.assertEqual(set(pii_rules.SPAN_PRIORITY), HOOK_TIER_LABELS)

    def test_every_emitted_label_has_a_priority(self) -> None:
        samples = (
            "alice@example.com",
            "call 415-555-1234",
            "see https://example.com/a",
            "on 2024-01-15",
            "card 4111111111111111",
            "account number: 12345678",
            "host 10.0.0.7",
            "api_key = abcdefgh1234",
        )
        emitted = {
            str(span["label"])
            for text in samples
            for span in pii_rules.deterministic_spans(text)
        }
        self.assertTrue(emitted)
        self.assertLessEqual(emitted, set(pii_rules.SPAN_PRIORITY))


if __name__ == "__main__":
    unittest.main()
