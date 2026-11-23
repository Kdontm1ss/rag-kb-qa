import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import chat
from app.bailian import ProviderError
from app.retrieval import ChunkHit


QUESTION = "年假怎么申请？"
NOTICE = "年假申请须提前三个工作日提交。"
APPROVAL = "请在考勤系统提交，由直属主管审批。"


def record(source: int, quote: str) -> str:
    return json.dumps({"source": source, "quote": quote}, ensure_ascii=False)


class ChatStreamTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.client = self.enterContext(TestClient(self.app))
        self.enterContext(patch.dict("os.environ", {"RERANK_MIN_SCORE": "0.25"}))
        contents = [f"其他资料{index}" for index in range(6)] + [APPROVAL, NOTICE]
        self.hits = [
            ChunkHit(
                chunk_id=index,
                document_id=UUID(int=index),
                title=f"制度{index}.pdf",
                content=content,
                chunk_index=index - 1,
                token_count=20,
                metadata={"page": index, "section": ["年假"]},
                score=0.03,
            )
            for index, content in enumerate(contents, 1)
        ]
        self.ranking = [(index, 0.99) for index in reversed(range(len(self.hits)))]

    def ask(self) -> list[dict]:
        response = self.client.post("/api/chat/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        self.assertTrue(events)
        self.assertTrue(events[-1]["done"])
        self.assertTrue(all(not event["done"] for event in events[:-1]))
        return events

    def test_reranks_eight_and_streams_only_top_four_sources_with_citations(self):
        async def generate(messages):
            sources = json.loads(messages[1]["content"])["sources"]
            self.assertEqual([source["text"] for source in sources], [hit.content for hit in reversed(self.hits[-4:])])
            first = record(1, NOTICE)
            yield first[:18]
            yield first[18:] + "\n"
            yield record(2, APPROVAL)

        with (
            patch.object(chat, "retrieve_hybrid", return_value=self.hits),
            patch.object(chat, "rerank", AsyncMock(return_value=self.ranking)) as rerank,
            patch.object(chat, "chat_deltas", generate),
        ):
            notice, approval, end = self.ask()

        rerank.assert_awaited_once_with(QUESTION, [hit.content for hit in self.hits])
        self.assertEqual(notice["answer_delta"], NOTICE + " [1]")
        self.assertEqual(approval["answer_delta"], "\n" + APPROVAL + " [2]")
        hit = self.hits[-1]
        self.assertEqual(notice["citations"], [{
            "number": 1, "document_name": hit.title, "document_id": str(hit.document_id),
            "chunk_id": hit.chunk_id, "chunk_index": hit.chunk_index, "page": 8, "section": ["年假"],
        }])
        self.assertEqual(end["status"], "answered")
        self.assertEqual(end["citations"], notice["citations"] + approval["citations"])

    def test_low_scores_finish_with_refusal_without_generation(self):
        with (
            patch.object(chat, "retrieve_hybrid", return_value=self.hits),
            patch.object(chat, "rerank", AsyncMock(return_value=[(i, 0.0) for i, _ in self.ranking])),
            patch.object(chat, "chat_deltas") as generate,
        ):
            refusal, = self.ask()

        generate.assert_not_called()
        self.assertEqual((refusal["status"], refusal["reason"]), ("refused", "low_relevance"))
        self.assertIn("暂无法回答", refusal["answer_delta"])
        self.assertEqual(refusal["citations"], [])

    def test_empty_search_finishes_without_reranking_or_generation(self):
        with (
            patch.object(chat, "retrieve_hybrid", return_value=[]),
            patch.object(chat, "rerank", AsyncMock()) as rerank,
            patch.object(chat, "chat_deltas") as generate,
        ):
            refusal, = self.ask()

        rerank.assert_not_called()
        generate.assert_not_called()
        self.assertEqual((refusal["status"], refusal["reason"]), ("refused", "no_evidence"))

    def test_refuses_when_the_model_has_no_evidence_or_the_quote_is_invalid(self):
        invalid = (
            ('{"refuse": true}', ""),
            (record(1, APPROVAL), APPROVAL),  # This quote belongs to source 2.
            (record(1, "年假提前五天提交。"), "年假提前五天提交。"),
        )
        for reply, rejected_quote in invalid:
            with self.subTest(reply=reply):
                async def generate(_messages):
                    yield reply + "\n"

                with (
                    patch.object(chat, "retrieve_hybrid", return_value=self.hits),
                    patch.object(chat, "rerank", AsyncMock(return_value=self.ranking)),
                    patch.object(chat, "chat_deltas", generate),
                ):
                    refusal, = self.ask()

                self.assertEqual((refusal["status"], refusal["reason"]), ("refused", "no_evidence"))
                self.assertEqual(refusal["citations"], [])
                if rejected_quote:
                    self.assertNotIn(rejected_quote, refusal["answer_delta"])

    def test_partial_answer_reports_upstream_failure_instead_of_success(self):
        async def generate(_messages):
            yield record(1, NOTICE) + "\n"
            raise ProviderError("upstream disconnected")

        with (
            patch.object(chat, "retrieve_hybrid", return_value=self.hits),
            patch.object(chat, "rerank", AsyncMock(return_value=self.ranking)),
            patch.object(chat, "chat_deltas", generate),
            self.assertLogs("app.chat", level="WARNING"),
        ):
            first, end = self.ask()

        self.assertEqual(first["answer_delta"], NOTICE + " [1]")
        self.assertEqual((end["status"], end["reason"]), ("error", "upstream_error"))
        self.assertEqual(end["citations"], first["citations"])

    def test_asgi_sends_delta_and_keepalive_before_upstream_finishes(self):
        async def exercise():
            first_sent = asyncio.Event()
            keepalive_sent = asyncio.Event()
            payloads = []

            async def generate(_messages):
                yield record(1, NOTICE) + "\n"
                await keepalive_sent.wait()
                yield record(2, APPROVAL) + "\n"

            async def receive():
                return {"type": "http.request", "body": json.dumps({"question": QUESTION}).encode()}

            async def send(message):
                body = message.get("body", b"")
                if body.startswith(b"data: "):
                    payloads.append(json.loads(body[6:]))
                    if len(payloads) == 1:
                        first_sent.set()
                if body.startswith(b": keepalive") and first_sent.is_set():
                    keepalive_sent.set()

            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": "/api/chat/ask", "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            }
            with (
                patch.object(chat, "retrieve_hybrid", return_value=self.hits),
                patch.object(chat, "rerank", AsyncMock(return_value=self.ranking)),
                patch.object(chat, "chat_deltas", generate),
                patch.object(chat, "KEEPALIVE_SECONDS", 0.01),
            ):
                await asyncio.wait_for(self.app(scope, receive, send), timeout=5)

            self.assertTrue(keepalive_sent.is_set())
            self.assertEqual([event["done"] for event in payloads], [False, False, True])
            self.assertEqual(payloads[-1]["status"], "answered")

        asyncio.run(exercise())
