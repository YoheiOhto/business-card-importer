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
        return result, f"{result['name']}\n全文"


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

        def register(value, attachment, context):
            saved.append(value)
            jpeg, filename, full_text = attachment
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))
            self.assertEqual(full_text, f"{value['name']}\n全文")
            if value["name"] == "佐藤 一郎":
                return "duplicate", "https://notion.so/old", "old-id"
            return "created", "https://notion.so/new", "new-id"

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

        def register(_value, _attachment, _context):
            raise AssertionError("dry run must not register")

        counts = self.run_sync(reader, register, dry_run=True)
        self.assertEqual((counts["read"], counts["failed"]), (1, 1))
        self.assertFalse((self.data / "synced.json").exists())
        self.assertIn("    会社名: 架空商事", self.lines)
        self.assertIn("      全文", self.lines)

        counts = self.run_sync(reader, lambda value, attachment, context: ("created", "https://notion.so/new", "new-id"))
        self.assertEqual((counts["created"], counts["failed"]), (1, 1))
        counts = self.run_sync(reader, lambda value, attachment, context: ("created", "https://notion.so/new", "new-id"))
        self.assertEqual((counts["skipped"], counts["failed"]), (1, 1))

    def test_settings_error_stops_and_uncertain_create_is_not_retried(self):
        self.image("a.jpg", "white")
        self.image("b.jpg", "gray")
        reader = FakeReader({"a.jpg": card(), "b.jpg": card(name="佐藤 一郎")})

        def broken(_value, _attachment, _context):
            raise CardError("notion.local.json に登録先のデータソース ID と列名を設定してください。", 503)

        counts = self.run_sync(reader, broken)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(len(reader.calls), 1)

        def lost(_value, _attachment, _context):
            raise CardError("登録結果を確認できませんでした。Notion で登録済みか確かめてから再試行してください。", 502)

        counts = self.run_sync(reader, lost)
        self.assertEqual(counts["uncertain"], 2)
        counts = self.run_sync(reader, lost)
        self.assertEqual(counts["skipped"], 2)

    def test_limit_reads_only_the_first_new_images(self):
        for index, color in enumerate(("white", "gray", "black")):
            self.image(f"{index}.jpg", color)
        reader = FakeReader({f"{index}.jpg": card(name=f"名前{index}") for index in range(3)})
        register = lambda value, attachment, context: ("created", "https://notion.so/new", "new-id")

        counts = self.run_sync(reader, register, limit=2)
        self.assertEqual((counts["created"], counts["left"]), (2, 1))
        counts = self.run_sync(reader, register, limit=2)
        self.assertEqual((counts["created"], counts["skipped"], counts["left"]), (1, 2, 0))
        self.assertEqual(reader.calls, ["0.jpg", "1.jpg", "2.jpg"])

    def test_rows_created_without_body_get_photo_and_text_once(self):
        self.image("a.jpg", "white")
        page = "41b227d008f181d2a577f0677a51d812"
        self.data.mkdir()
        from sync import file_sha256
        ledger = {file_sha256(self.inbox / "a.jpg"): {
            "file": "a.jpg", "status": "created", "url": f"https://app.notion.com/p/{page}"}}
        (self.data / "synced.json").write_text(json.dumps(ledger), encoding="utf-8")
        reader = FakeReader({"a.jpg": card()})
        attached = []

        def register(_value, _attachment, _context):
            raise AssertionError("existing rows must not be created again")

        def attach(page_id, attachment):
            attached.append((page_id, attachment[1], attachment[2]))

        counts = self.run_sync(reader, register, attach=attach)
        self.assertEqual(counts["attached"], 1)
        self.assertEqual(attached, [(page, "a.jpg", "山田 花子\n全文")])
        counts = self.run_sync(reader, register, attach=attach)
        self.assertEqual((counts["skipped"], len(attached)), (1, 1))

    def test_event_and_date_go_to_new_rows_and_label_existing_ones_once(self):
        self.image("a.jpg", "white")
        reader = FakeReader({"a.jpg": card(), "b.jpg": card(name="佐藤 一郎")})
        given = []

        def register(_value, _attachment, context):
            given.append(context)
            return "created", "https://notion.so/41b227d008f181d2a577f0677a51d812", "page-a"

        tagged = []
        tag = lambda page_id, context: tagged.append((page_id, context))
        self.run_sync(reader, register, tag=tag)
        self.assertEqual(given, [{}])

        self.image("b.jpg", "gray")
        event = {"event": "CBI学会2026", "met_on": "2026-11-14"}
        counts = self.run_sync(reader, register, tag=tag, context=event)
        self.assertEqual((counts["created"], counts["tagged"]), (1, 1))
        self.assertEqual(given[-1], event)
        self.assertEqual(tagged, [("page-a", event)])

        counts = self.run_sync(reader, register, tag=tag, context=event)
        self.assertEqual((counts["tagged"], len(tagged)), (0, 1))
        counts = self.run_sync(reader, register, tag=tag, context={"event": "別の学会"})
        self.assertEqual(counts["tagged"], 2)
        self.assertEqual(tagged[-1][1], {"event": "別の学会"})


class CheckNotionTest(unittest.TestCase):
    def test_check_reads_schema_and_reports_wrong_column(self):
        import httpx
        from functools import partial
        from unittest.mock import patch

        from app import check_notion

        schema = {label: {"type": kind} for label, kind in FIELDS.values()}
        methods = []

        def notion(request):
            methods.append(request.method)
            return httpx.Response(200, json={"properties": schema})

        factory = partial(httpx.Client, transport=httpx.MockTransport(notion))
        environment = {"NOTION_TOKEN": "test", "NOTION_DATA_SOURCE_ID": "0b4595fd-a02d-4b71-95c5-0c482316cc45"}
        with patch("app.httpx.Client", side_effect=factory), patch.dict("os.environ", environment):
            columns, present = check_notion()
            self.assertEqual(present, {"event": False, "met_on": False})
            schema["メール"] = {"type": "rich_text"}
            with self.assertRaisesRegex(CardError, "メール"):
                check_notion()
        self.assertEqual(methods, ["GET", "GET"])


class AttachmentTest(unittest.TestCase):
    def test_save_card_uploads_photo_and_writes_full_text(self):
        import httpx
        from functools import partial
        from unittest.mock import patch

        from app import save_card

        schema = {label: {"type": kind} for label, kind in FIELDS.values()}
        calls = []

        def notion(request):
            calls.append((request.method, request.url.path))
            if request.method == "GET":
                return httpx.Response(200, json={"properties": schema})
            if request.url.path == "/v1/file_uploads":
                body = json.loads(request.content)
                self.assertEqual(body["content_type"], "image/jpeg")
                return httpx.Response(200, json={"id": "upload-1", "status": "pending"})
            if request.url.path == "/v1/file_uploads/upload-1/send":
                self.assertIn(b"JPEGDATA", request.content)
                self.assertTrue(request.headers["content-type"].startswith("multipart/form-data"))
                return httpx.Response(200, json={"id": "upload-1", "status": "uploaded"})
            body = json.loads(request.content)
            image, text = body["children"][1], body["children"][3]
            self.assertEqual(image["image"]["file_upload"]["id"], "upload-1")
            self.assertEqual("".join(part["text"]["content"] for part in text["paragraph"]["rich_text"]),
                             "山田 花子\n" + "あ" * 2500)
            self.assertEqual(len(text["paragraph"]["rich_text"]), 2)
            return httpx.Response(200, json={"id": "page-1", "url": "https://notion.so/page-1"})

        factory = partial(httpx.Client, transport=httpx.MockTransport(notion))
        environment = {"NOTION_TOKEN": "test", "NOTION_DATA_SOURCE_ID": "0b4595fd-a02d-4b71-95c5-0c482316cc45"}
        with patch("app.httpx.Client", side_effect=factory), patch.dict("os.environ", environment):
            result = save_card(card(), (b"JPEGDATA", "a.jpg", "山田 花子\n" + "あ" * 2500))
        self.assertEqual(result, ("created", "https://notion.so/page-1", "page-1"))
        self.assertEqual([path for _, path in calls], [
            "/v1/data_sources/0b4595fd-a02d-4b71-95c5-0c482316cc45",
            "/v1/file_uploads", "/v1/file_uploads/upload-1/send", "/v1/pages"])


class ContextPropertyTest(unittest.TestCase):
    def test_select_date_and_missing_column(self):
        from app import context_properties

        columns = {"event": "学会", "met_on": "日付"}
        schema = {"学会": {"type": "select"}, "日付": {"type": "date"}}
        self.assertEqual(
            context_properties(schema, columns, {"event": "CBI学会2026", "met_on": "2026-11-14"}),
            {"学会": {"select": {"name": "CBI学会2026"}}, "日付": {"date": {"start": "2026-11-14"}}})
        self.assertEqual(context_properties(schema, columns, None), {})
        with self.assertRaisesRegex(CardError, "日付"):
            context_properties({}, columns, {"met_on": "2026-11-14"})


class ParseTest(unittest.TestCase):
    def test_parse_model_answer(self):
        text = '```json\n{"full_text": "山田 花子\n架空商事", "name": " 山田 花子 ", "email": null}\n```'
        self.assertEqual(parse_response(text), (card(), "山田 花子\n架空商事"))
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
