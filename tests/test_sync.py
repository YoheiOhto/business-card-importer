import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import FIELDS, CardError
from sync import clean_card, parse_response, sync


def card(**overrides):
    return dict.fromkeys(FIELDS, "") | {"name": "山田 花子"} | overrides


class FakeReader:
    def __init__(self, cards):
        self.cards = cards
        self.calls = []

    def read(self, path, digest):
        self.calls.append(path.name)
        result = self.cards[path.name]
        if isinstance(result, Exception):
            raise result
        return result


class SyncTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.inbox = Path(folder.name) / "inbox"
        self.data = Path(folder.name) / "data"
        self.inbox.mkdir()
        self.lines = []

    def image(self, name, color):
        Image.new("RGB", (40, 24), color).save(self.inbox / name)

    def run_sync(self, reader, register, **options):
        return sync(self.inbox, self.data, reader, register=register, out=self.lines.append, **options)

    def test_registers_new_images_once_and_skips_them_next_time(self):
        self.image("a.jpg", "white")
        self.image("b.png", "gray")
        self.image("copy-of-a.jpg", "white")  # Same bytes as a.jpg.
        (self.inbox / "notes.txt").write_text("not an image")
        reader = FakeReader({
            "a.jpg": card(email="hanako@example.com"),
            "b.png": card(name="佐藤 一郎", email="ichiro@example.com"),
        })
        saved = []

        def register(value):
            saved.append(value)
            if value["name"] == "佐藤 一郎":
                return "duplicate", "https://notion.so/old"
            return "created", "https://notion.so/new"

        counts = self.run_sync(reader, register)
        self.assertEqual((counts["created"], counts["duplicate"], counts["skipped"]), (1, 1, 1))
        self.assertEqual([value["name"] for value in saved], ["山田 花子", "佐藤 一郎"])
        ledger = json.loads((self.data / "synced.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(entry["status"] for entry in ledger.values()), ["created", "duplicate"])

        counts = self.run_sync(reader, register)
        self.assertEqual(counts["skipped"], 3)
        self.assertEqual(len(saved), 2)

    def test_dry_run_sends_nothing_and_failed_reads_are_retried(self):
        self.image("a.jpg", "white")
        self.image("blank.jpg", "black")
        reader = FakeReader({"a.jpg": card(company="架空商事"), "blank.jpg": card(name="")})

        def register(_value):
            raise AssertionError("dry run must not register")

        counts = self.run_sync(reader, register, dry_run=True)
        self.assertEqual((counts["read"], counts["failed"]), (1, 1))
        self.assertFalse((self.data / "synced.json").exists())
        self.assertIn("    会社名: 架空商事", self.lines)

        counts = self.run_sync(reader, lambda value: ("created", "https://notion.so/new"))
        self.assertEqual((counts["created"], counts["failed"]), (1, 1))
        counts = self.run_sync(reader, lambda value: ("created", "https://notion.so/new"))
        self.assertEqual((counts["skipped"], counts["failed"]), (1, 1))

    def test_settings_error_stops_and_uncertain_create_is_not_retried(self):
        self.image("a.jpg", "white")
        self.image("b.jpg", "gray")
        reader = FakeReader({"a.jpg": card(), "b.jpg": card(name="佐藤 一郎")})

        def broken(_value):
            raise CardError("notion.local.json に登録先のデータソース ID と列名を設定してください。", 503)

        counts = self.run_sync(reader, broken)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(len(reader.calls), 1)

        def lost(_value):
            raise CardError("登録結果を確認できませんでした。Notion で登録済みか確かめてから再試行してください。", 502)

        counts = self.run_sync(reader, lost)
        self.assertEqual(counts["uncertain"], 2)
        counts = self.run_sync(reader, lost)
        self.assertEqual(counts["skipped"], 2)


class ParseTest(unittest.TestCase):
    def test_parse_model_answer(self):
        text = '```json\n{"name": " 山田 花子 ", "email": null, "extra": "x"}\n```'
        self.assertEqual(parse_response(text), card())
        with self.assertRaises(CardError):
            parse_response("読めませんでした")

    def test_invalid_optional_fields_are_dropped(self):
        value, dropped = clean_card(card(email="hanako at example", website="example.com", phone="03-0000-0000"))
        self.assertEqual(dropped, ["メール"])
        self.assertEqual((value["email"], value["website"]), ("", "https://example.com"))
        self.assertEqual(value["phone"], "03-0000-0000")
        with self.assertRaises(CardError):
            clean_card(card(name=""))


if __name__ == "__main__":
    unittest.main()
