import json
import os
from collections.abc import AsyncIterator

import httpx

CHAT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INFERENCE_TIMEOUT = httpx.Timeout(45, connect=5)


class ProviderError(Exception):
    pass


def provider_settings() -> tuple[str, dict[str, str]]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ProviderError("请配置 DASHSCOPE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
    return base_url, {"Authorization": f"Bearer {api_key}"}


async def rerank(question: str, documents: list[str]) -> list[tuple[int, float]]:
    if not documents:
        return []

    chat_base, headers = provider_settings()
    try:
        base_url = os.getenv("RERANK_BASE_URL") or httpx.URL(chat_base).copy_with(
            path="/api/v1", query=None, fragment=None
        )
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=INFERENCE_TIMEOUT) as client:
            response = await client.post(
                "services/rerank/text-rerank/text-rerank",
                json={
                    "model": os.getenv("RERANK_MODEL") or "qwen3-rerank",
                    "input": {"query": question, "documents": documents},
                    "parameters": {"top_n": len(documents), "return_documents": False},
                },
            )
            response.raise_for_status()
            rows = response.json()["output"]["results"]
            ranks = [(row["index"], row["relevance_score"]) for row in rows]
            if any(type(index) is not int or type(score) not in (int, float) or not 0 <= score <= 1
                   for index, score in ranks):
                raise ValueError("invalid reranker result")
            if len(ranks) != len(documents) or {index for index, _ in ranks} != set(range(len(documents))):
                raise ValueError("incomplete reranker result")
    except httpx.HTTPStatusError as error:
        raise ProviderError(f"百炼重排服务返回 HTTP {error.response.status_code}") from error
    except (httpx.RequestError, httpx.InvalidURL) as error:
        raise ProviderError(f"无法调用百炼重排服务 ({type(error).__name__})") from error
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderError("百炼重排服务响应格式无效") from error

    return sorted(ranks, key=lambda result: (-result[1], result[0]))


async def sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    data_fields = []
    async for line in response.aiter_lines():
        if not line and data_fields:
            yield "\n".join(data_fields)
            data_fields.clear()
        elif line.startswith("data:"):
            data_fields.append(line[len("data:"):].removeprefix(" "))


async def chat_deltas(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    if os.getenv("LLM_MODEL", CHAT_MODEL) != CHAT_MODEL:
        raise ProviderError("LLM_MODEL 必须为 qwen-plus")

    base_url, headers = provider_settings()
    complete = False
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=INFERENCE_TIMEOUT) as client:
            async with client.stream(
                "POST",
                "chat/completions",
                json={
                    "model": CHAT_MODEL,
                    "messages": messages,
                    "stream": True,
                    "enable_thinking": False,
                    "temperature": 0.1,
                    "max_tokens": 1600,
                },
            ) as response:
                response.raise_for_status()
                async for payload in sse_payloads(response):
                    if payload == "[DONE]":
                        if complete:
                            return
                        break

                    event = json.loads(payload)
                    choices = event["choices"]
                    if choices == []:
                        continue
                    choice, = choices
                    if complete or choice["index"] != 0:
                        raise ValueError("unexpected completion choice")
                    reason = choice["finish_reason"]
                    if reason not in (None, "stop"):
                        raise ProviderError("百炼 LLM 未完整生成答案")
                    complete = reason == "stop"

                    content = choice["delta"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise ValueError("invalid completion content")
                    if content:
                        yield content
    except httpx.HTTPStatusError as error:
        raise ProviderError(f"百炼 LLM 返回 HTTP {error.response.status_code}") from error
    except (httpx.RequestError, httpx.InvalidURL) as error:
        raise ProviderError(f"无法调用百炼 LLM ({type(error).__name__})") from error
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ProviderError("百炼 LLM 流式响应格式无效") from error

    raise ProviderError("百炼 LLM 流提前中断")
