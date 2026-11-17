import base64
import io
import json
import unittest
from functools import partial
from unittest.mock import patch

import httpx
from PIL import Image

from app import FIELDS, MODEL, app

SOURCE_ID = "0b4595fd-a02d-4b71-95c5-0c482316cc45"
PAGE_URL = "https://www.notion.so/4e383131d3f44d2b989568f96824e22f"
HEADER = {"X-Card-Request": "1"}


def sample_card(**overrides):
    return dict.fromkeys(FIELDS, "") | {"name": "山田 花子"} | overrides


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.browser = app.test_client()

    def api(self, handler):
        factory = partial(httpx.Client, transport=httpx.MockTransport(handler))
        return patch("app.httpx.Client", side_effect=factory)

    def test_read_image_with_local_model(self):
        output = sample_card(company="架空商事")
        buffer = io.BytesIO()
        Image.new("RGB", (32, 20), "white").save(buffer, "PNG")

        def ollama(request):
            self.assertEqual(str(request.url), "http://127.0.0.1:11434/api/chat")
            body = json.loads(request.content)
            self.assertEqual(body["model"], MODEL)
            self.assertEqual(body["format"]["required"], list(FIELDS))
            payload = base64.b64decode(body["messages"][0]["images"][0])
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.size, (32, 20))
            return httpx.Response(200, json={"message": {"content": json.dumps(output)}})

        with self.api(ollama), patch.dict("os.environ", {"CARD_OLLAMA_PORT": "11434"}):
            response = self.browser.post("/api/extract", data=buffer.getvalue(), headers=HEADER)
        self.assertEqual(response.get_json(), {"card": output})

    def test_register_edited_fields_or_show_duplicate(self):
        card = sample_card(email="hanako@example.com", company="修正後の会社名", website="example.com")
        columns = {field: label for field, (label, _) in FIELDS.items()}
        columns["company"] = "所属先"
        schema = {columns[field]: {"type": kind} for field, (_, kind) in FIELDS.items()}

        for exists in (False, True):
            with self.subTest(duplicate=exists):
                calls = []

                def notion(request):
                    calls.append(request.url.path)
                    self.assertEqual(request.url.host, "api.notion.com")
                    self.assertEqual(request.headers["Notion-Version"], "2026-03-11")
                    if request.method == "GET":
                        self.assertEqual(request.url.path, f"/v1/data_sources/{SOURCE_ID}")
                        return httpx.Response(200, json={"properties": schema})
                    body = json.loads(request.content)
                    if request.url.path.endswith("/query"):
                        self.assertEqual(body["filter"], {
                            "property": "メール", "email": {"equals": "hanako@example.com"}
                        })
                        return httpx.Response(200, json={"results": [{"url": PAGE_URL}] if exists else []})
                    self.assertEqual(request.url.path, "/v1/pages")
                    self.assertEqual(body["parent"], {"type": "data_source_id", "data_source_id": SOURCE_ID})
                    self.assertEqual(set(body), {"parent", "properties"})
                    properties = body["properties"]
                    self.assertEqual(set(properties), {"氏名", "所属先", "メール", "Webサイト"})
                    self.assertEqual(properties["所属先"]["rich_text"][0]["text"]["content"], "修正後の会社名")
                    self.assertEqual(properties["Webサイト"], {"url": "https://example.com"})
                    return httpx.Response(200, json={"id": "mock-id", "url": PAGE_URL})

                with (
                    self.api(notion),
                    patch("app.read_notion_config", return_value=(SOURCE_ID, columns)),
                    patch("app.notion_token", return_value="synthetic-token"),
                ):
                    response = self.browser.post("/api/register", json=card, headers=HEADER)
                self.assertEqual(response.status_code, 409 if exists else 201)
                self.assertEqual(response.get_json()["url"], PAGE_URL)
                self.assertEqual(len(calls), 2 if exists else 3)

    def test_invalid_image_blank_name_and_cross_origin_request(self):
        response = self.browser.post("/api/extract", data=b"invalid", headers=HEADER)
        self.assertEqual(response.status_code, 400)
        response = self.browser.post("/api/register", json=sample_card(name="  "), headers=HEADER)
        self.assertEqual(response.status_code, 400)
        response = self.browser.post("/api/register", json=sample_card())
        self.assertEqual(response.status_code, 403)
        response = self.browser.get("/", headers={"Host": "external.example"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
