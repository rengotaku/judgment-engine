"""OpenAI 互換の Chat Completions を教師として使うアダプタ。

DeepSeek・OpenAI・ローカルの vLLM など、`/v1/chat/completions` を話すものなら同じ実装で動く。
**この経路は課金が発生する。** CLI からは `--allow-paid` を明示しないと動かない。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from .base import LabelResult, SampleRequest, TeacherError, validate_labels

PROMPT_VERSION = "teacher-json-v1"

SYSTEM_PROMPT = (
    "You label decision tasks. For each task, read `state` and `question`, then return the "
    "single best answer from the allowed answer space. Answer only with JSON: "
    '{"labels":[{"id":"<id>","label":"<answer>"}]}. Do not add explanations. '
    "If the state does not contain enough information, still pick the most likely answer."
)


def allowed_labels(question: dict[str, Any]) -> list[str]:
    """質問から取りうる答えを列挙する（choice のキー / yes,no / score のレベル番号）。"""
    qtype = question.get("type")
    if qtype == "choice":
        return list((question.get("criteria") or {}).keys())
    if qtype == "score":
        return [str(index) for index in range(len(question.get("criteria") or []))]
    if qtype == "noul":
        return ["yes", "no"]
    raise TeacherError(f"未知の type: {qtype!r}")


def build_payload(
    requests: Sequence[SampleRequest], model: str, prompt_version: str = PROMPT_VERSION
) -> dict[str, Any]:
    tasks = [
        {
            "id": request.id,
            "state": request.state,
            "question": request.question,
            "allowed": allowed_labels(request.question),
        }
        for request in requests
    ]
    return {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"prompt_version": prompt_version, "tasks": tasks}, ensure_ascii=False),
            },
        ],
    }


def parse_response(content: str, requests: Sequence[SampleRequest]) -> list[LabelResult]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TeacherError(f"教師の応答が JSON ではありません: {exc}") from exc
    items = payload.get("labels")
    if not isinstance(items, list):
        raise TeacherError("教師の応答に labels がありません")

    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            by_id[item["id"]] = item

    results: list[LabelResult] = []
    for request in requests:
        item = by_id.get(request.id)
        if item is None:
            raise TeacherError(f"{request.id}: 教師の応答に含まれていません")
        label = str(item.get("label", ""))
        if label not in allowed_labels(request.question):
            raise TeacherError(f"{request.id}: 答えの範囲外です: {label!r}")
        results.append(LabelResult(label=label, raw=json.dumps(item, ensure_ascii=False)))
    validate_labels(requests, results)
    return results


class OpenAICompatibleTeacher:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        batch_size: int = 10,
        timeout_sec: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout_sec = timeout_sec
        self.name = f"openai_compatible:{model}"
        self.prompt_version = PROMPT_VERSION

    @classmethod
    def from_env(cls, model: str, base_url: str, **kwargs: Any) -> OpenAICompatibleTeacher:
        api_key = os.environ.get("JE_TEACHER_API_KEY", "")
        if not api_key:
            raise TeacherError("JE_TEACHER_API_KEY がありません（実教師は課金されます）")
        return cls(model=model, base_url=base_url, api_key=api_key, **kwargs)

    def label(self, requests: Sequence[SampleRequest]) -> list[LabelResult]:
        import httpx

        results: list[LabelResult] = []
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout_sec) as client:
            for start in range(0, len(requests), self.batch_size):
                chunk = list(requests[start : start + self.batch_size])
                payload = build_payload(chunk, self.model, self.prompt_version)
                try:
                    response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise TeacherError(f"教師の呼び出しに失敗しました: {exc}") from exc
                body = response.json()
                try:
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise TeacherError(f"教師の応答の形が想定と違います: {body}") from exc
                results.extend(parse_response(content, chunk))
        return results
