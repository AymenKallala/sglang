"""Exercise model I/O processors through the Python HTTP serving routes."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import math
import sys
import unittest
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.plugins.io_processor import PreparedInput, load_io_processor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class LengthReranker:
    def __init__(self, *, tokenizer, hf_config):
        pass

    def prepare(self, data):
        doc = data["document"]
        return PreparedInput(data["query"] + " " + doc, len(doc))

    def postprocess(self, row, context):
        return row[0] / context


class TestIOProcessors(CustomTestCase):
    def setUp(self):
        tokenizer = Tokenizer(
            models.WordLevel({"?": 0, "yes": 1, "no": 2, "<end>": 3}, unk_token="?")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<end>",
            unk_token="?",
            chat_template="{% for m in messages %}{{ m.content }} {{ eos_token }} {% endfor %}",
        )
        self.hf_config = SimpleNamespace(id2label={0: "positive", 1: "negative"})
        self.tm = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(hf_config=self.hf_config),
            server_args=None,
            served_model_name="classifier",
            context_len=8,
            request_logger=SimpleNamespace(log_requests=False),
        )
        self.app = FastAPI()
        self.app.router.routes.extend(
            route
            for route in http_server.app.routes
            if getattr(route, "path", None) in ("/v1/classify", "/v1/rerank")
        )

    def install(self, task, name, value):
        self.hf_config.sglang_io_processors = {task: name}
        entry = EntryPoint(name=name, value=value, group="sglang.io_processors")
        other = EntryPoint(
            name="unused", value="missing_module:Unused", group=entry.group
        )
        self.enterContext(
            patch("sglang.srt.plugins.entry_points", return_value=[entry, other])
        )

    def client(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    def test_conversation_scores(self):
        # Load the example factory through its package entry point.
        example = Path(__file__).resolve().parents[5] / "examples/runtime/io_processor"
        self.enterContext(patch.object(sys, "path", [str(example), *sys.path]))
        self.install("classify", "turn_scores", "turn_scores:TurnScores")
        self.app.state.openai_serving_classify = OpenAIServingClassify(self.tm, None)
        both_ready = asyncio.Event()
        active = []

        async def generate(req, raw_request):
            if req.text:
                yield {"embedding": [0.0, 1.0], "meta_info": {"prompt_tokens": 1}}
                return
            active.append(req.input_ids)
            if len(active) == 2:
                both_ready.set()
            await asyncio.wait_for(both_ready.wait(), timeout=5)
            yield {
                "embedding": [[i, -i] for i in range(len(req.input_ids))],
                "meta_info": {"prompt_tokens": len(req.input_ids)},
            }

        self.tm.generate_request = generate

        async def run():
            async with self.client() as client:

                async def classify(messages):
                    return await client.post(
                        "/v1/classify", json={"input": {"messages": messages}}
                    )

                first = {"role": "user", "content": "yes"}
                second = {"role": "assistant", "content": "no"}
                a, b = await asyncio.gather(
                    classify([first]), classify([first, second])
                )
                for response, positions in ((a, [1]), (b, [1, 3])):
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    turns = body["data"][0]["turns"]
                    self.assertEqual(len(turns), len(positions))
                    self.assertEqual(body["usage"]["prompt_tokens"], positions[-1] + 1)
                    for turn, pos in zip(turns, positions, strict=True):
                        self.assertAlmostEqual(
                            turn["positive"], 1 / (1 + math.exp(-pos))
                        )
                        self.assertAlmostEqual(
                            turn["negative"], 1 / (1 + math.exp(pos))
                        )
                standard = await client.post("/v1/classify", json={"input": "yes"})
                self.assertEqual(standard.json()["data"][0]["label"], "negative")
                for messages in ([], [first] * 5):
                    response = await classify(messages)
                    self.assertEqual(response.status_code, 400, response.text)
                self.assertCountEqual(active, [[1, 3], [1, 3, 2, 3]])

        asyncio.run(run())

    def test_unconfigured_object_input_and_missing_plugin(self):
        self.app.state.openai_serving_classify = OpenAIServingClassify(self.tm, None)

        async def run():
            async with self.client() as client:
                response = await client.post(
                    "/v1/classify", json={"input": {"messages": []}}
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("I/O processor", response.json()["message"])

        asyncio.run(run())
        self.hf_config.sglang_io_processors = {"classify": "missing"}
        with (
            patch("sglang.srt.plugins.entry_points", return_value=[]),
            self.assertRaisesRegex(ValueError, "missing"),
        ):
            load_io_processor("classify", self.tm)

    def test_rerank_processor_batches_and_preserves_ranking_options(self):
        self.install("rerank", "length", f"{__name__}:LengthReranker")
        self.tm.score_prompts = AsyncMock(
            return_value=SimpleNamespace(scores=[[0.8], [0.7]])
        )
        self.app.state.openai_serving_rerank = OpenAIServingRerank(self.tm)

        async def run():
            async with self.client() as client:
                return await client.post(
                    "/v1/rerank",
                    json={
                        "query": "Q",
                        "documents": ["long", "a"],
                        "top_n": 1,
                        "return_documents": False,
                    },
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()[0]["index"], 1)
        self.assertAlmostEqual(response.json()[0]["score"], 0.7)
        self.assertNotIn("document", response.json()[0])
        self.assertEqual(len(response.json()), 1)
        args, kwargs = self.tm.score_prompts.await_args
        self.assertEqual(args, (["Q long", "Q a"],))
        self.assertIsNone(kwargs["label_token_ids"])


if __name__ == "__main__":
    unittest.main()
