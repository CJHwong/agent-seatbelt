"""Unit tests for the rules that run after the bilingual tagger.

Ported from the training repository's suite, so the hook keeps the scored behaviour.
Standard library only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from rules.tagger_rules import (
    apply_chain,
    office_filter,
    place_filter,
    post_rules,
    public_filter,
    username_keys,
)

PUBLIC = {"歐巴馬", "barack obama", "蔡英文"}


def _span(text: str, part: str, label: str) -> dict:
    start = text.index(part)
    return {"start": start, "end": start + len(part), "label": label, "score": 0.9}


def _persons(text: str, *names: str) -> list[dict]:
    spans, start = ([], 0)
    for name in names:
        start = text.index(name, start)
        spans.append(
            {"start": start, "end": start + len(name), "label": "person", "score": 0.9}
        )
        start += len(name)
    return spans


def _spans_of(text: str, label: str, *values: str) -> list[dict]:
    spans, start = ([], 0)
    for value in values:
        start = text.index(value, start)
        spans.append(
            {"start": start, "end": start + len(value), "label": label, "score": 0.9}
        )
        start += len(value)
    return spans


def _values(spans, text):
    return [(text[s["start"] : s["end"]], s["label"]) for s in spans]


class TaggerRulesTests(unittest.TestCase):
    def test_post_rules_drop_a_chinese_city_or_district_alone_but_keep_street_addresses(
        self,
    ) -> None:
        text = "他住在台北市大安區，公司在台北市信義區松仁路100號5樓，辦公室在C座，老家在 Jalan Muhibbah."
        spans = [
            _span(text, "台北市大安區", "address"),
            _span(text, "台北市信義區松仁路100號5樓", "address"),
            _span(text, "C座", "address"),
            _span(text, "Jalan Muhibbah", "address"),
        ]
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in post_rules(spans, text)],
            ["台北市信義區松仁路100號5樓", "C座", "Jalan Muhibbah"],
        )

    def test_post_rules_drop_public_host_urls_but_keep_profiles_and_userinfo(
        self,
    ) -> None:
        text = "see https://docs.python.org/3/ and https://example.com/a, me at https://github.com/users/ann or https://bob:pw@pypi.org/x"
        spans = [
            _span(text, "https://docs.python.org/3/", "url"),
            _span(text, "https://example.com/a", "url"),
            _span(text, "https://github.com/users/ann", "url"),
            _span(text, "https://bob:pw@pypi.org/x", "url"),
        ]
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in post_rules(spans, text)],
            ["https://github.com/users/ann", "https://bob:pw@pypi.org/x"],
        )

    def test_post_rules_drop_shared_mailboxes_and_default_accounts(self) -> None:
        text = "mail service@shop.tw, ann@example.org or ann.lee@gmail.com; login root or annlee88"
        spans = [
            _span(text, "service@shop.tw", "email"),
            _span(text, "ann@example.org", "email"),
            _span(text, "ann.lee@gmail.com", "email"),
            _span(text, "root", "username"),
            _span(text, "annlee88", "username"),
        ]
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in post_rules(spans, text)],
            ["ann@example.org", "ann.lee@gmail.com", "annlee88"],
        )

    def test_post_rules_drop_a_username_inside_a_dropped_url(self) -> None:
        text = "clone https://pypi.org/project/requests now"
        spans = [
            _span(text, "https://pypi.org/project/requests", "url"),
            _span(text, "requests", "username"),
        ]
        self.assertEqual(post_rules(spans, text), [])

    def test_public_filter_drops_a_listed_full_name_and_its_later_surname(self) -> None:
        text = "Barack Obama spoke today. Later Obama met 蔡英文 and Ann Lee."
        spans = [
            _span(text, "Barack Obama", "person"),
            _span(text, "Obama met", "person") | {"end": text.index("Obama met") + 5},
            _span(text, "蔡英文", "person"),
            _span(text, "Ann Lee", "person"),
        ]
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in public_filter(spans, text, PUBLIC)],
            ["Ann Lee"],
        )

    def test_public_filter_keeps_a_listed_name_next_to_other_pii(self) -> None:
        text = "蔡英文 的手機 0912345678"
        spans = [_span(text, "蔡英文", "person"), _span(text, "0912345678", "phone")]
        self.assertEqual(public_filter(spans, text, PUBLIC), spans)

    def test_public_filter_ignores_a_single_word_or_character_match(self) -> None:
        text = "Obama and 蔡 came"
        spans = [_span(text, "Obama", "person"), _span(text, "蔡", "person")]
        self.assertEqual(public_filter(spans, text, PUBLIC | {"obama", "蔡"}), spans)

    def test_place_filter_drops_person_spans_that_name_a_listed_place(self) -> None:
        text = "德國與菲律賓的代表王小明，以及 Fiji 和 New Zealand"
        spans = [
            _span(text, "德國", "person"),
            _span(text, "菲律賓", "person"),
            _span(text, "王小明", "person"),
            _span(text, "Fiji", "person"),
            _span(text, "New Zealand", "person"),
        ]
        places = {"德國", "菲律賓", "fiji", "new zealand"}
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in place_filter(spans, text, places)],
            ["王小明", "Fiji", "New Zealand"],
        )

    def test_office_filter_drops_a_titled_name_and_its_later_mentions(self) -> None:
        text = "立法會議員郭榮鏗今日表示反對。郭榮鏗又說，陳怡君的看法不同。"
        spans = _persons(text, "郭榮鏗", "郭榮鏗", "陳怡君")
        kept = office_filter(spans, text)
        self.assertEqual([text[s["start"] : s["end"]] for s in kept], ["陳怡君"])

    def test_office_filter_reads_titles_after_the_name_and_english_titles(self) -> None:
        text = "彭斯副總統 met Senator John Smith and jake."
        spans = _persons(text, "彭斯", "John Smith", "jake")
        self.assertEqual(
            [text[s["start"] : s["end"]] for s in office_filter(spans, text)], ["jake"]
        )

    def test_office_filter_keeps_a_titled_name_next_to_private_pii(self) -> None:
        text = "議員王小明 電話0912345678"
        spans = _persons(text, "王小明") + [
            {"start": 9, "end": 19, "label": "phone", "score": 0.9}
        ]
        self.assertEqual(office_filter(spans, text), spans)

    def test_post_rules_drop_code_identifiers_flagged_as_usernames_but_keep_handles(
        self,
    ) -> None:
        text = "x = nx.Graph()\nhidden_size = 4\nassert_frame_equal(a, b)\nclass Foo:\nmail from jeanette.blair and my venmo is jake_99"
        spans = _spans_of(
            text,
            "username",
            "nx.Graph",
            "hidden_size",
            "assert_frame_equal",
            "Foo",
            "jeanette.blair",
            "jake_99",
        )
        kept = [text[s["start"] : s["end"]] for s in post_rules(spans, text)]
        self.assertEqual(kept, ["jeanette.blair", "jake_99"])

    def test_post_rules_drop_repository_and_documentation_urls_but_keep_profiles(
        self,
    ) -> None:
        text = "see https://github.com/huggingface/transformers/issues/1 and http://www.apache.org/licenses/LICENSE-2.0 or docs/source/en/fusion_mapping.md; my page is https://github.com/lumina37 and https://www.pixnet.net/blog/me"
        spans = _spans_of(
            text,
            "url",
            "https://github.com/huggingface/transformers/issues/1",
            "http://www.apache.org/licenses/LICENSE-2.0",
            "docs/source/en/fusion_mapping.md",
            "https://github.com/lumina37",
            "https://www.pixnet.net/blog/me",
        )
        kept = [text[s["start"] : s["end"]] for s in post_rules(spans, text)]
        self.assertEqual(
            kept, ["https://github.com/lumina37", "https://www.pixnet.net/blog/me"]
        )

    def test_username_keys_label_a_handle_after_a_username_key(self) -> None:
        roster = "- { name: Jo Lee, slack: U011ABCDE9, github: jlee-42 }"
        log = "PR#12 | author=mchen | reviewer=t.wu | state=open"
        self.assertEqual(
            _values(username_keys([], roster), roster),
            [("U011ABCDE9", "username"), ("jlee-42", "username")],
        )
        self.assertEqual(
            _values(username_keys([], log), log),
            [("mchen", "username"), ("t.wu", "username")],
        )

    def test_username_keys_read_table_mentions_but_not_paths_or_account_keys(
        self,
    ) -> None:
        paths = (
            "gh api /users/octocat/repos; ls /Users/jdoe/Documents and /home/deploy/app"
        )
        accounts = "login: dbuser\nusername: svc_backup\nhandle: ops"
        table = "| Jo Lee | @jlee42 | backend |"
        self.assertEqual(username_keys([], paths), [])
        self.assertEqual(username_keys([], accounts), [])
        self.assertEqual(
            _values(username_keys([], table), table), [("jlee42", "username")]
        )

    def test_username_keys_skip_placeholders_bots_and_covered_values(self) -> None:
        skipped = "author=<login> slack: TBD github: null author=dependabot[bot] reviewer=renovate-bot author: 12345"
        self.assertEqual(username_keys([], skipped), [])
        covered = "author=mchen"
        existing = [{"start": 7, "end": 12, "label": "person", "score": 0.9}]
        self.assertEqual(username_keys(existing, covered), existing)

    def test_username_keys_skip_namespaced_commands_calls_code_and_review_bots(
        self,
    ) -> None:
        command = "run /gf-github:pr-create then `tool-gitlab:sync`"
        query = "assignee = currentUser() AND status = Done; assignee=currentUser()"
        code = "author = 'PlaintextProbe'\nreviewer=reviewer"
        bots = "reviewer=human reviewer=coderabbit author: copilot-pull-request-reviewer assignee=unassigned | @reviewer |"
        for text in (command, query, code, bots):
            self.assertEqual(username_keys([], text), [], text)

    def test_apply_chain_cuts_unsure_urls_then_adds_field_handles(self) -> None:
        text = "see https://www.pixnet.net/blog/me | author=mchen"
        spans = [{"start": 4, "end": 34, "label": "url", "score": 0.4}]
        kept = apply_chain(spans, text, set(), set(), {"url": 0.5})
        self.assertEqual(_values(kept, text), [("mchen", "username")])


if __name__ == "__main__":
    unittest.main()
