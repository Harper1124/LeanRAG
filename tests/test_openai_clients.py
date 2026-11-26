import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from multimodal.openai_clients import (
    make_async_chat_func,
    make_chat_func,
    make_embedding_func,
)


class _Response:
    choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]


class _AsyncCompletions:
    active = 0
    max_active = 0

    async def create(self, **_kwargs):
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        await asyncio.sleep(0.01)
        type(self).active -= 1
        return _Response()


class _FakeAsyncOpenAI:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=_AsyncCompletions())


class _FakeOpenAI:
    init_kwargs = []

    def __init__(self, **kwargs):
        type(self).init_kwargs.append(kwargs)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace())
        self.embeddings = types.SimpleNamespace()


class OpenAIClientConfigurationTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.init_kwargs = []
        _FakeAsyncOpenAI.init_kwargs = None
        _AsyncCompletions.active = 0
        _AsyncCompletions.max_active = 0
        self.openai_module = types.SimpleNamespace(
            OpenAI=_FakeOpenAI,
            AsyncOpenAI=_FakeAsyncOpenAI,
        )

    def test_sync_clients_receive_timeout_and_retry_configuration(self):
        config = {
            "model": "model",
            "embedding_model": "embedding",
            "api_key": "key",
            "base_url": "http://localhost/v1",
            "timeout": 123,
            "max_retries": 4,
        }
        with patch.dict(sys.modules, {"openai": self.openai_module}):
            make_chat_func(config)
            make_embedding_func(config)

        self.assertEqual(len(_FakeOpenAI.init_kwargs), 2)
        for kwargs in _FakeOpenAI.init_kwargs:
            self.assertEqual(kwargs["timeout"], 123.0)
            self.assertEqual(kwargs["max_retries"], 4)

    def test_async_client_serializes_requests_when_max_concurrency_is_one(self):
        config = {
            "model": "model",
            "api_key": "key",
            "base_url": "http://localhost/v1",
            "timeout": 321,
            "max_retries": 5,
            "max_concurrency": 1,
        }
        with patch.dict(sys.modules, {"openai": self.openai_module}):
            chat = make_async_chat_func(config)

        async def run_calls():
            return await asyncio.gather(chat("one"), chat("two"), chat("three"))

        self.assertEqual(asyncio.run(run_calls()), ["ok", "ok", "ok"])
        self.assertEqual(_AsyncCompletions.max_active, 1)
        self.assertEqual(_FakeAsyncOpenAI.init_kwargs["timeout"], 321.0)
        self.assertEqual(_FakeAsyncOpenAI.init_kwargs["max_retries"], 5)


if __name__ == "__main__":
    unittest.main()
