"""Behavior tests for the deterministic PII rule engine.

Every test asserts the span offsets, labels or merged text that the module
returns. Nothing here mocks the rules. The engine tests swap in a stand-in for
the native module, because that is the only way to reach its fallback paths.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import types
import unicodedata
import unittest
from pathlib import Path
from typing import cast
from unittest import mock


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


def labeled_texts(text: str) -> set[tuple[str, str]]:
    """Return (text, label) for each deterministic span."""
    return {
        (str(span["text"]), str(span["label"]))
        for span in pii_rules.deterministic_spans(text)
    }


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


TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from challenge_cases import CASES as CHALLENGE_CASES  # noqa: E402
from final_holdout_cases import FINAL_HOLDOUT_CASES  # noqa: E402
from holdout_cases import HOLDOUT_CASES  # noqa: E402


def corpus_texts() -> list[tuple[str, str]]:
    """Every case text in the repository, named so a failure says which one."""
    texts = []
    for case in [*CHALLENGE_CASES, *HOLDOUT_CASES, *FINAL_HOLDOUT_CASES]:
        texts.append((str(case["id"]), str(case["text"])))
    for number, line in enumerate(
        (TESTS_DIR / "test-cases.jsonl").read_text().splitlines()
    ):
        texts.append((f"test-cases:{number}", json.loads(line)["prompt"]))
    for line in (TESTS_DIR / "false-positive-cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        texts.append((case["id"], case["text"]))
    # The rule source itself: long, dense with keywords, and holding CJK text.
    texts.append(("pii_rules.py", Path(pii_rules.__file__).read_text()))
    for sample in PORTED_SAMPLES:
        texts.append((f"sample:{sample['rule']}", sample["text"]))
    return texts


# One generated value per ported rule, placed in a line of code, one file per
# source.
# The values are random strings shaped by each rule, not real credentials.
PORTED_SAMPLES = [
    json.loads(line)
    for source in ("gitleaks", "betterleaks", "microsoft", "seatbelt")
    for line in (TESTS_DIR / f"{source}-samples.jsonl").read_text().splitlines()
]


def only_ported_rule(rule_id: str):
    """Scan with one ported rule as the only secret rule."""
    pattern = pii_rules.PORTED_PATTERNS[rule_id]
    checks = tuple(c for c in pii_rules.PORTED_CHECKS if c[0] is pattern)
    return mock.patch.multiple(
        pii_rules, SECRET_RULES=(), PORTED_CHECKS=checks, _NATIVE_ENGINE=None
    )


class RuleFalsePositiveTests(unittest.TestCase):
    """Clean text the rules once flagged, next to the secret each rule is for."""

    def test_the_rules_flag_none_of_these_clean_cases(self) -> None:
        cases = {
            case["id"]: case["text"]
            for case in map(
                json.loads,
                (TESTS_DIR / "false-positive-cases.jsonl").read_text().splitlines(),
            )
        }
        # The URL rule reports every URL, a public one included, by design.
        for case_id in [f"clean-{number:03}" for number in range(51, 58)]:
            with self.subTest(case=case_id):
                labels = {
                    span["label"]
                    for span in pii_rules.deterministic_spans(cases[case_id])
                }
                self.assertLessEqual(labels, {"private_url"})

    def test_a_secret_keyword_joined_to_a_word_is_not_a_secret_context(self) -> None:
        self.assertEqual(span_texts("docs/code-security/secret-scanning/alerts"), set())
        link = "https://onetimesecret.com/secret/4f9k2m8q1x7w3z6b5n0p"
        self.assertIn(("/4f9k2m8q1x7w3z6b5n0p", "secret"), labeled_texts(link))
        self.assertEqual(span_texts("secret: q8Rv2LmX7pWz4NbK"), {"q8Rv2LmX7pWz4NbK"})
        self.assertEqual(span_texts("secret q8Rv2LmX7pWz4NbK"), {"q8Rv2LmX7pWz4NbK"})

    def test_one_quoted_name_per_line_is_not_a_csv_header(self) -> None:
        self.assertEqual(span_texts('    "password",\n    "hunter2value",\n'), set())
        self.assertEqual(
            span_texts("password,\nhunter2value,\n"),
            set(),
        )

    def test_a_generic_context_value_of_lowercase_words_is_not_a_secret(self) -> None:
        self.assertEqual(span_texts("airtable-api-key: keyword-context rule"), set())
        self.assertEqual(span_texts("api-key: kx82mzq0dl"), {"kx82mzq0dl"})
        # A password key still reads a passphrase of words, quoted or not.
        for text in (
            "password: correct-horse-battery-staple",
            '"password": "correct-horse-battery-staple"',
            '{"db_password": "correct-horse-battery-staple"}',
        ):
            with self.subTest(text=text):
                self.assertEqual(span_texts(text), {"correct-horse-battery-staple"})


class PortedRuleTests(unittest.TestCase):
    def test_every_rule_finds_its_sample(self) -> None:
        self.assertEqual(
            {sample["rule"] for sample in PORTED_SAMPLES},
            set(pii_rules.PORTED_PATTERNS),
        )
        for sample in PORTED_SAMPLES:
            with self.subTest(rule=sample["rule"]), only_ported_rule(sample["rule"]):
                self.assertIn(sample["value"], span_texts(sample["text"]))

    def test_a_secret_at_or_under_the_entropy_floor_is_not_reported(self) -> None:
        # The rule's floor is 2 bits. "ab" repeated holds about 1.9.
        text = "rubygems_" + "ab" * 24 + "\n"
        with only_ported_rule("rubygems-api-token"):
            self.assertEqual(span_texts(text), set())
            self.assertEqual(
                span_texts("rubygems_" + "0123456789abcdef" * 3 + "\n"),
                {"rubygems_" + "0123456789abcdef" * 3},
            )

    def test_a_secret_that_tokenizes_like_words_is_not_reported(self) -> None:
        # 60 characters in 14 tokens is a ratio of 4.3, over the ceiling of 2.5.
        words = "vcp_the_production_deployment_token_for_our_marketing_webapp"
        with only_ported_rule("betterleaks/vercel-personal-access-token"):
            self.assertEqual(span_texts(f'token = "{words}"'), set())
            random = "vcp_" + "Kx8mQ2vLp9ZrT4wNc7YhB3" * 2 + "q0Wn7Rt4Bz9L"
            self.assertEqual(span_texts(f'token = "{random}"'), {random})

    def test_a_prefix_inside_a_base64_run_is_not_a_key(self) -> None:
        # A long base64 blob, an embedded image, holds any short prefix sooner
        # or later. "/" is a word boundary, but it is a base64 character.
        key = "AKLT" + "Kx8mQ2vLp9ZrT4wNc7YhB3q0Wn7Rt4Bz9L"
        with only_ported_rule("seatbelt/volcengine-access-key-id"):
            self.assertEqual(span_texts(f"iVBORw0KGgo+Qm9/{key}/x9Tq+Lm2"), set())
            self.assertEqual(span_texts(f"VOLC_ACCESSKEY={key}\n"), {key})

    def test_an_allowlisted_secret_is_not_reported(self) -> None:
        # gitleaks lists this key as a known sample value.
        text = 'key: "AIzaSyabcdefghijklmnopqrstuvwxyz1234567"'
        with only_ported_rule("gcp-api-key"):
            self.assertEqual(span_texts(text), set())

    def test_the_ported_rules_add_nothing_to_the_other_cases(self) -> None:
        for name, text in corpus_texts():
            if name.startswith("sample:"):
                continue
            with self.subTest(case=name), python_engine():
                with mock.patch.object(pii_rules, "PORTED_CHECKS", ()):
                    without = pii_rules.deterministic_spans(text)
                self.assertEqual(pii_rules.deterministic_spans(text), without)

    def test_the_github_checklist_names_only_real_rules(self) -> None:
        rows = [
            line.split("\t")
            for line in (TESTS_DIR / "github-secret-types.tsv").read_text().splitlines()
            if not line.startswith("#")
        ]
        self.assertEqual(
            rows[0], ["secret_type", "provider", "push_protection", "rule_id"]
        )
        named = {row[3] for row in rows[1:]} - {"-"}
        rule_patterns = {name for name in vars(pii_rules) if name.endswith("_PATTERN")}
        # Every ported rule covers a type, and every name is a rule.
        self.assertLessEqual(set(pii_rules.PORTED_PATTERNS), named)
        self.assertLessEqual(named, set(pii_rules.PORTED_PATTERNS) | rule_patterns)


class TokenCountTests(unittest.TestCase):
    def test_the_vocabulary_keeps_every_rank(self) -> None:
        ranks = pii_rules._cl100k_ranks()
        self.assertEqual(sorted(ranks.values()), list(range(100256)))
        # Ranks from OpenAI's tiktoken.
        self.assertEqual(ranks[b"hello"], 15339)
        self.assertEqual(ranks[b" world"], 1917)

    def test_the_count_matches_the_tokenizer_betterleaks_uses(self) -> None:
        # Counted with tiktoken-go v0.1.8, cl100k_base, as betterleaks counts.
        # OpenAI's tiktoken gives the same counts.
        expected = {
            "hello world": 2,
            "vcp_the_production_deployment_token_for_our_marketing_webapp": 14,
            "Kx8mQ2vLp9ZrT4wNc7YhB3": 22,
            "0123456789abcdef": 5,
            "it's we'll THEY'RE": 6,
            "a  b\n\n  c   ": 7,
            "naïve café 東京タワー": 11,
            "__init__.py -- ++==": 7,
            "½ ² Ⅻ 3.14159": 11,
            "": 0,
        }
        for text, count in expected.items():
            with self.subTest(text=text):
                self.assertEqual(pii_rules._token_count(text), count)


def python_engine():
    return mock.patch.object(pii_rules, "_NATIVE_ENGINE", None)


class KeywordGateTests(unittest.TestCase):
    """A gate may only skip a pattern that could not have matched."""

    def test_the_gates_never_change_a_result(self) -> None:
        # Ported keywords are rule semantics, not literals the pattern needs.
        ported = set(pii_rules.PORTED_PATTERNS.values())
        ungated = {
            pattern: keywords if pattern in ported else ()
            for pattern, keywords in pii_rules.RULE_KEYWORDS.items()
        }
        for name, text in corpus_texts():
            with self.subTest(case=name), python_engine():
                gated_spans = pii_rules.deterministic_spans(text)
                with mock.patch.object(pii_rules, "RULE_KEYWORDS", ungated):
                    self.assertEqual(gated_spans, pii_rules.deterministic_spans(text))

    def test_a_folding_character_opens_every_gate(self) -> None:
        # IGNORECASE matches the long s to "s", so "ſecret" is the secret
        # keyword to re, while a search for "secret" does not find it.
        text = "\u017fecret: q8Rv2LmX7pWz4NbK"
        with python_engine():
            self.assertIn(
                ("q8Rv2LmX7pWz4NbK", "secret"),
                [
                    (str(span["text"]), str(span["label"]))
                    for span in pii_rules.deterministic_spans(text)
                ],
            )

    def test_every_secret_column_name_holds_a_keyword(self) -> None:
        for name in pii_rules.SECRET_COLUMN_NAMES:
            with self.subTest(name=name):
                self.assertTrue(
                    any(k in name for k in pii_rules.SECRET_COLUMN_KEYWORDS)
                )

    def test_a_header_that_folds_to_a_secret_column_is_still_read(self) -> None:
        # casefold turns "PAßWORD" into "password", and lower() does not.
        text = "user,PA\u00dfWORD\nada,q8Rv2LmX7pWz\n"
        self.assertIn("q8Rv2LmX7pWz", span_texts(text))

    def test_every_pattern_has_an_entry(self) -> None:
        not_rules = ("INVISIBLE_PATTERN",)
        declared = {
            value
            for name, value in vars(pii_rules).items()
            if name.endswith("_PATTERN") and name not in not_rules
        }
        declared.update(pii_rules.PORTED_PATTERNS.values())
        self.assertEqual(declared, set(pii_rules.RULE_KEYWORDS))


@unittest.skipUnless(
    pii_rules.RULES_ENGINE == "native", "the native engine is not built here"
)
class NativeEngineTests(unittest.TestCase):
    """The native engine must return exactly what re returns."""

    def native_scan(self, text: str):
        engine = pii_rules._NATIVE_ENGINE
        assert engine is not None
        return engine.scan(text)

    def assert_same_as_python(self, text: str) -> None:
        native_spans = pii_rules.deterministic_spans(text)
        with python_engine():
            self.assertEqual(native_spans, pii_rules.deterministic_spans(text))

    def test_native_matches_python_on_every_corpus_text(self) -> None:
        for name, text in corpus_texts():
            with self.subTest(case=name):
                self.assert_same_as_python(text)

    def test_offsets_count_code_points_not_bytes(self) -> None:
        text = "\u6d4b\u8bd5 \U0001f600 mail ada@example.com"
        self.assertEqual(span_tuples(text), [(10, 25, "private_email")])
        self.assert_same_as_python(text)

    def test_a_combining_mark_is_matched_the_way_re_matches_it(self) -> None:
        # PCRE2's own \w counts U+0301 and U+FE0F as word characters and re
        # does not, so these guards only agree because \w is translated.
        for text in (
            "cafe\u0301ada@example.com",
            "\u2764\ufe0fada@example.com",
            "ok\u0301 AKIAABCDEFGHIJKLMNOP\u0301",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(self.native_scan(text))
                self.assert_same_as_python(text)

    def test_a_character_the_unicode_tables_disagree_on_is_declined(self) -> None:
        import pii_rules_native  # ty: ignore[unresolved-import]

        decline = pii_rules._native_decline_characters(pii_rules_native)
        newer = [c for c in decline if c not in pii_rules.FOLDING_CHARACTERS]
        if not newer:
            self.skipTest("Python and PCRE2 share one Unicode version here")
        text = f"{newer[0]}1234567890 ada@example.com"
        self.assertIsNone(self.native_scan(text))
        self.assert_same_as_python(text)

    def test_pcre2_word_characters_differ_only_where_the_engine_checks(self) -> None:
        """PCRE2's own \\w must equal re's everywhere but PCRE2_WORD_DIFFERENCE.

        A text without those characters is matched with PCRE2's \\w and \\b,
        so any other difference would reach a result.
        """
        import pii_rules_native  # ty: ignore[unresolved-import]

        every = "".join(
            map(chr, itertools.chain(range(0xD800), range(0xE000, 0x110000)))
        )
        members = pii_rules_native.class_members
        own = set(members(r"\w", False, every))
        spelled = set(members(f"[{pii_rules.PYTHON_WORD_CHARACTERS}]", False, every))
        checked = set(members(pii_rules.PCRE2_WORD_DIFFERENCE, False, every))
        self.assertEqual(own ^ spelled, checked)

    def test_the_tables_only_disagree_on_characters_python_does_not_know(self) -> None:
        """The derivation must find version drift and nothing else.

        Apart from the folding characters and the controls the strip removes,
        a character the two engines place differently must be one this
        Python's Unicode tables leave unassigned.
        """
        import pii_rules_native  # ty: ignore[unresolved-import]

        derived = pii_rules._derive_decline_characters(pii_rules_native.class_members)
        unexplained = [
            f"U+{ord(c):04X}"
            for c in derived
            if c not in pii_rules.FOLDING_CHARACTERS and unicodedata.category(c) != "Cn"
        ]
        self.assertEqual(unexplained, [])
        self.assertTrue(all(not c.isascii() for c in derived))

    def test_a_folding_character_is_declined_and_still_scanned(self) -> None:
        text = "\u017fecret: q8Rv2LmX7pWz4NbK"
        self.assertIsNone(self.native_scan(text))
        self.assertIn("q8Rv2LmX7pWz4NbK", span_texts(text))

    def test_a_lone_surrogate_is_scanned_by_re(self) -> None:
        text = "\ud800 ada@example.com"
        self.assertIn("ada@example.com", span_texts(text))
        self.assert_same_as_python(text)

    def test_an_invisible_character_is_stripped_like_python_strips_it(self) -> None:
        text = "ada@exa\u200bmple.com and 4111\u00ad1111\u00ad1111\u00ad1111"
        self.assertIn("ada@exa\u200bmple.com", span_texts(text))
        self.assert_same_as_python(text)


class PatternTranslationTests(unittest.TestCase):
    """The native engine reads \\w and \\b the way re does."""

    word = f"[{pii_rules.PYTHON_WORD_CHARACTERS}]"

    def test_a_word_class_outside_a_set_becomes_a_set(self) -> None:
        self.assertEqual(pii_rules._pcre2_source(r"a\wb"), f"a{self.word}b")

    def test_a_word_class_inside_a_set_joins_it(self) -> None:
        self.assertEqual(
            pii_rules._pcre2_source(r"(?<![\w+-])"),
            f"(?<![{pii_rules.PYTHON_WORD_CHARACTERS}+-])",
        )

    def test_a_word_boundary_becomes_lookarounds(self) -> None:
        self.assertEqual(
            pii_rules._pcre2_source(r"\bAKIA"), pii_rules.PYTHON_WORD_BOUNDARY + "AKIA"
        )

    def test_an_escaped_backslash_is_left_alone(self) -> None:
        self.assertEqual(pii_rules._pcre2_source(r"\\w\\b"), r"\\w\\b")

    def test_a_class_with_no_translation_is_refused(self) -> None:
        for source in (r"\W", r"\B"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    pii_rules._pcre2_source(source)


class UnicodeVersionTests(unittest.TestCase):
    """The derivation is skipped only when both tables are the same version."""

    def test_the_same_version_declines_only_the_folding_characters(self) -> None:
        def never(*_arguments):
            raise AssertionError("derived although the versions match")

        module = types.SimpleNamespace(
            UNICODE_VERSION=unicodedata.unidata_version, class_members=never
        )
        self.assertEqual(
            set(pii_rules._native_decline_characters(module)),
            set(pii_rules.FOLDING_CHARACTERS),
        )

    def test_another_version_derives_the_characters(self) -> None:
        calls = []

        def members(pattern, ignore_case, text):
            calls.append(pattern)
            flags = re.IGNORECASE if ignore_case else 0
            python = {r"[\p{L}\p{N}_]": r"\w"}.get(pattern, pattern)
            return "".join(re.findall(python, text, flags))

        module = types.SimpleNamespace(UNICODE_VERSION="0.0.0", class_members=members)
        self.assertEqual(
            set(pii_rules._native_decline_characters(module)),
            set(pii_rules.FOLDING_CHARACTERS),
        )
        self.assertEqual(len(calls), len(pii_rules.UNICODE_CLASS_CHECKS))


class StandInEngine:
    """A native engine that fails in one chosen way."""

    def __init__(self, scan_result=None, scan_error=None) -> None:
        self.scan_result = scan_result
        self.scan_error = scan_error

    def scan(self, text: str):
        if self.scan_error is not None:
            raise self.scan_error
        return self.scan_result

    def strip_invisibles(self, text: str):
        return None


class EngineFallbackTests(unittest.TestCase):
    """Whatever goes wrong with the native module, re still scans the text."""

    text = "mail ada@example.com"
    expected = [(5, 20, "private_email")]

    def load_with(self, module) -> tuple[object | None, str]:
        with mock.patch.dict(sys.modules, {"pii_rules_native": module}):
            return pii_rules._load_native_engine()

    def test_a_missing_module_names_the_reason(self) -> None:
        engine, reason = self.load_with(None)
        self.assertIsNone(engine)
        self.assertTrue(reason.startswith("python ("), reason)

    def test_a_module_of_another_version_is_ignored(self) -> None:
        module = types.SimpleNamespace(
            ENGINE_VERSION=pii_rules.NATIVE_ENGINE_VERSION + 1
        )
        engine, reason = self.load_with(module)
        self.assertIsNone(engine)
        self.assertIn("these rules need version", reason)

    def test_a_module_that_refuses_a_pattern_is_ignored(self) -> None:
        def refuse(*_arguments):
            raise ValueError("unsupported construct")

        module = types.SimpleNamespace(
            ENGINE_VERSION=pii_rules.NATIVE_ENGINE_VERSION,
            UNICODE_VERSION=unicodedata.unidata_version,
            Engine=refuse,
        )
        engine, reason = self.load_with(module)
        self.assertIsNone(engine)
        self.assertIn("unsupported construct", reason)

    def test_a_declined_text_is_scanned_by_re(self) -> None:
        with mock.patch.object(pii_rules, "_NATIVE_ENGINE", StandInEngine()):
            self.assertEqual(span_tuples(self.text), self.expected)

    def test_a_scan_error_is_scanned_by_re(self) -> None:
        errors = (
            ValueError("match limit exceeded"),
            UnicodeEncodeError("utf-8", "", 0, 1, "surrogate"),
        )
        for error in errors:
            stand_in = StandInEngine(scan_error=error)
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(pii_rules, "_NATIVE_ENGINE", stand_in):
                    self.assertEqual(span_tuples(self.text), self.expected)


if __name__ == "__main__":
    unittest.main()
