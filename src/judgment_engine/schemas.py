"""Jev 互換のリクエスト/レスポンス定義。

Jev（TypeSafe System One）の形に合わせる:

    POST /v1/systemone
    {
      "model": "jev-latest",
      "state": "...",
      "questions": {
        "department": {"type": "choice", "instructions": "...",
                       "criteria": {"returns": "...", "billing": "..."}},
        "needs_human": {"type": "noul", "instructions": "..."},
        "severity":   {"type": "score", "instructions": "...", "criteria": ["低", "中", "高"]}
      }
    }

独自拡張は `routing` と任意の `policy` だけ。`answers` の形は Jev と同じにして、
既存クライアント（`answers.<id>.noul` などを読むもの）を壊さない。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Limits

QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
InstructionValue = str | dict[str, Any] | list[Any]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    instructions: InstructionValue


class NoulQuestion(QuestionBase):
    type: Literal["noul"]
    # {"true": "...", "false": "..."}。Jev では省略可。
    criteria: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> NoulQuestion:
        if self.criteria is None:
            return self
        unknown = set(self.criteria) - {"true", "false"}
        if unknown:
            raise ValueError(f"noul の criteria は true/false のみ: {sorted(unknown)}")
        return self


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    # 値は None（説明なし）・文字列・構造化（dict / list）を許す。
    criteria: dict[str, Any]


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    # 並び順がレベルの番号（0 起点）。
    criteria: list[Any]


class Policy(BaseModel):
    """任意の拡張。省略時はサーバ既定で routing を決める。"""

    model_config = ConfigDict(extra="forbid")

    abstain_below: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    # 高リスクは human へ回す。Phase 0 は判断材料を受け取るだけ。
    risk: Literal["low", "high"] | None = None
    allow_teacher: bool = True


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]]
    model: str | None = None
    policy: Policy | None = None
    request_id: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(max_length=128)
    question_id: str = Field(max_length=64)
    # 正解・訂正・実行結果のどれか（自由記述のラベル）
    outcome: Literal["correct", "incorrect", "executed_ok", "executed_failed", "unknown"]
    label: str | None = Field(default=None, max_length=512)
    note: str | None = Field(default=None, max_length=2_000)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


class Routing(BaseModel):
    route: Literal["local", "teacher", "human"]
    reason: str
    confidence: float
    margin: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated: bool = True
    latency_ms: float
    pairs: int


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    routing: dict[str, Routing]
    usage: Usage
    request_id: str
    backend: str


def parse_questions(raw: dict[str, dict[str, Any]], limits: Limits) -> dict[str, QuestionBase]:
    """`type` を見て各質問を検証する。上限違反は ValueError（422 にする）。"""
    if not raw:
        raise ValueError("questions が空です")
    if len(raw) > limits.max_questions:
        raise ValueError(f"questions は {limits.max_questions} 件までです")

    parsed: dict[str, QuestionBase] = {}
    for qid, body in raw.items():
        if not QUESTION_ID_RE.match(qid):
            raise ValueError(f"question id は英数字と _.:- の 1〜{limits.max_question_id_chars} 文字: {qid!r}")
        qtype = body.get("type")
        if qtype == "noul":
            question: QuestionBase = NoulQuestion.model_validate(body)
        elif qtype == "choice":
            question = ChoiceQuestion.model_validate(body)
        elif qtype == "score":
            question = ScoreQuestion.model_validate(body)
        else:
            raise ValueError(f"未知の type: {qtype!r}（noul / choice / score）")
        _check_instructions(question.instructions, limits, qid)
        if isinstance(question, ChoiceQuestion):
            _check_choice(question.criteria, limits, qid)
        elif isinstance(question, ScoreQuestion):
            _check_score(question.criteria, limits, qid)
        parsed[qid] = question
    return parsed


def _text_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(len(str(k)) + _text_length(v) for k, v in value.items())
    if isinstance(value, list):
        return sum(_text_length(v) for v in value)
    return len(str(value))


def _check_instructions(value: InstructionValue, limits: Limits, qid: str) -> None:
    length = _text_length(value)
    if length == 0:
        raise ValueError(f"{qid}: instructions が空です")
    if length > limits.max_instructions_chars:
        raise ValueError(f"{qid}: instructions が長すぎます（{length} > {limits.max_instructions_chars}）")


def _check_choice(criteria: dict[str, Any], limits: Limits, qid: str) -> None:
    if not (limits.min_options <= len(criteria) <= limits.max_options):
        raise ValueError(
            f"{qid}: choice の候補は {limits.min_options}〜{limits.max_options} 件です（{len(criteria)} 件）"
        )
    for key, value in criteria.items():
        if not key or len(key) > limits.max_option_key_chars:
            raise ValueError(f"{qid}: 候補名は 1〜{limits.max_option_key_chars} 文字です: {key!r}")
        if value is not None and _text_length(value) > limits.max_option_text_chars:
            raise ValueError(f"{qid}: 候補 {key} の説明が長すぎます（上限 {limits.max_option_text_chars}）")


def _check_score(criteria: list[Any], limits: Limits, qid: str) -> None:
    if not (limits.min_score_levels <= len(criteria) <= limits.max_score_levels):
        raise ValueError(
            f"{qid}: score のレベルは {limits.min_score_levels}〜{limits.max_score_levels} 件です（{len(criteria)} 件）"
        )
    for index, value in enumerate(criteria):
        if _text_length(value) > limits.max_option_text_chars:
            raise ValueError(f"{qid}: レベル {index} の説明が長すぎます（上限 {limits.max_option_text_chars}）")
