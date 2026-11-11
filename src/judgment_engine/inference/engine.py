"""判断の中核。質問を組に分解し、スコアラの出力を型付きの確率へ変換する。

`state` と `questions` から作る文字列は、このファイルだけが決める。評価ハーネスも
HTTP を経由せず同じ関数を通すので、API と評価で前処理がズレない。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Policy,
    QuestionBase,
    Routing,
    ScoreAnswer,
    ScoreQuestion,
)
from .base import Scorer

# テンプレートはここで固定する。変えたら評価の基準値も取り直す。
CONTEXT_TEMPLATE = "State:\n{state}\n\nQuestion: {instructions}"
CHOICE_CANDIDATE_TEMPLATE = "Option: {key}\n{description}"
NOUL_CANDIDATES = ("Answer: yes", "Answer: no")
SCORE_CANDIDATE_TEMPLATE = "Level {index}: {description}"


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def render_instructions(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False, separators=(",", ":"))


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def choice_pairs(state: str, question: ChoiceQuestion) -> tuple[list[tuple[str, str]], list[str]]:
    context = CONTEXT_TEMPLATE.format(state=state, instructions=render_instructions(question.instructions))
    keys = list(question.criteria)
    pairs = [
        (context, CHOICE_CANDIDATE_TEMPLATE.format(key=key, description=render_value(question.criteria[key])))
        for key in keys
    ]
    return pairs, keys


def noul_pairs(state: str, question: NoulQuestion) -> tuple[list[tuple[str, str]], list[str]]:
    criteria = question.criteria or {}
    context_parts = [CONTEXT_TEMPLATE.format(state=state, instructions=render_instructions(question.instructions))]
    if criteria:
        context_parts.append(
            "Yes means: {yes}\nNo means: {no}".format(
                yes=render_value(criteria.get("true", "")) or "the statement holds",
                no=render_value(criteria.get("false", "")) or "the statement does not hold",
            )
        )
    context = "\n\n".join(context_parts)
    return [(context, candidate) for candidate in NOUL_CANDIDATES], ["yes", "no"]


def score_pairs(state: str, question: ScoreQuestion) -> tuple[list[tuple[str, str]], list[int]]:
    context = CONTEXT_TEMPLATE.format(state=state, instructions=render_instructions(question.instructions))
    indexes = list(range(len(question.criteria)))
    pairs = [
        (context, SCORE_CANDIDATE_TEMPLATE.format(index=index, description=render_value(question.criteria[index])))
        for index in indexes
    ]
    return pairs, indexes


def softmax(logits: Sequence[float], temperature: float) -> list[float]:
    if not logits:
        return []
    if temperature <= 0:
        raise ValueError("temperature は正の数にしてください")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def confidence_of(probabilities: Sequence[float]) -> float:
    """Jev の confidence と同じ式: (p_max - 1/n) / (1 - 1/n)。

    公式ドキュメントの例（0.61/0.35/0.04 → 0.42、0.84/0.16/0 → 0.76、
    0.4/0.34/0.24/0.02 → 0.20、0.76/0.24/0 → 0.64）と一致することを確認済み。
    """
    if not probabilities:
        return 0.0
    if len(probabilities) == 1:
        return 1.0
    top = max(probabilities)
    return max(0.0, min(1.0, (top - 1.0 / len(probabilities)) / (1.0 - 1.0 / len(probabilities))))


def margin_of(probabilities: Sequence[float]) -> float:
    ordered = sorted(probabilities, reverse=True)
    if len(ordered) < 2:
        return 1.0
    return ordered[0] - ordered[1]


@dataclass(frozen=True)
class QuestionResult:
    answer: NoulAnswer | ChoiceAnswer | ScoreAnswer
    routing: Routing
    # 上位確率（noul は max(p, 1-p)）— 評価の coverage 判定に使う
    top_probability: float
    margin: float


class JudgmentEngine:
    def __init__(self, scorer: Scorer, settings: Settings) -> None:
        self.scorer = scorer
        self.settings = settings

    def count_pairs(self, questions: Iterable[QuestionBase], state: str) -> int:
        total = 0
        for question in questions:
            if isinstance(question, ChoiceQuestion):
                total += len(question.criteria)
            elif isinstance(question, ScoreQuestion):
                total += len(question.criteria)
            else:
                total += len(NOUL_CANDIDATES)
        return total

    def decide(
        self,
        state: Any,
        questions: dict[str, QuestionBase],
        policy: Policy | None = None,
    ) -> tuple[dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], dict[str, Routing], int, int]:
        """1 リクエスト分をまとめて処理する。返り値は (answers, routing, pairs, tokens)。"""
        state_text = render_state(state)
        if len(state_text) > self.settings.limits.max_state_chars:
            raise ValueError(
                f"state が長すぎます（{len(state_text)} > {self.settings.limits.max_state_chars}）"
            )

        # 全質問の組をまとめてスコアラへ渡す（質問は並列に評価される）
        all_pairs: list[tuple[str, str]] = []
        question_pairs: dict[str, list[tuple[str, str]]] = {}
        question_keys: dict[str, list[Any]] = {}
        for qid, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                pairs, keys = choice_pairs(state_text, question)
            elif isinstance(question, ScoreQuestion):
                pairs, keys = score_pairs(state_text, question)
            else:
                pairs, keys = noul_pairs(state_text, question)
            question_pairs[qid] = pairs
            question_keys[qid] = keys
            all_pairs.extend(pairs)

        logits = self.scorer.score_pairs(all_pairs) if all_pairs else []
        cursor = 0
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        routing: dict[str, Routing] = {}

        for qid, question in questions.items():
            count = len(question_pairs[qid])
            chunk = logits[cursor : cursor + count]
            cursor += count
            keys = question_keys[qid]
            probabilities = softmax(chunk, self.settings.temperature)

            if isinstance(question, NoulQuestion):
                p_yes = probabilities[0] if probabilities else 0.0
                answer: NoulAnswer | ChoiceAnswer | ScoreAnswer = NoulAnswer(noul=p_yes)
                top_probability = max(p_yes, 1.0 - p_yes)
                margin = abs(p_yes - 0.5) * 2.0
                confidence = top_probability
            elif isinstance(question, ChoiceQuestion):
                by_key = dict(zip(keys, probabilities, strict=True))
                best = max(by_key, key=by_key.__getitem__)
                answer = ChoiceAnswer(choice=best, confidence=confidence_of(probabilities), probabilities=by_key)
                top_probability = by_key[best]
                margin = margin_of(probabilities)
                confidence = answer.confidence
            else:
                by_level = {str(index): probability for index, probability in zip(keys, probabilities, strict=True)}
                expected = sum(index * probability for index, probability in zip(keys, probabilities, strict=True))
                legend = {str(index): question.criteria[index] for index in keys}
                answer = ScoreAnswer(
                    score=expected,
                    confidence=confidence_of(probabilities),
                    legend=legend,
                    probabilities=by_level,
                )
                top_probability = max(probabilities) if probabilities else 0.0
                margin = margin_of(probabilities)
                confidence = answer.confidence

            answers[qid] = answer
            routing[qid] = self._route(confidence, margin, policy)

        tokens = self.scorer.input_tokens(all_pairs) if all_pairs else 0
        return answers, routing, len(all_pairs), tokens

    def _route(self, confidence: float, margin: float, policy: Policy | None) -> Routing:
        abstain_below = (
            policy.abstain_below if policy and policy.abstain_below is not None else self.settings.abstain_below
        )
        minimum_margin = (
            policy.minimum_margin if policy and policy.minimum_margin is not None else self.settings.minimum_margin
        )
        allow_teacher = policy.allow_teacher if policy else True

        if policy and policy.risk == "high":
            return Routing(route="human", reason="high_risk", confidence=confidence, margin=margin)
        if confidence < abstain_below:
            return Routing(
                route="teacher" if allow_teacher else "human",
                reason="confidence_below_threshold",
                confidence=confidence,
                margin=margin,
            )
        if margin < minimum_margin:
            return Routing(
                route="teacher" if allow_teacher else "human",
                reason="margin_below_threshold",
                confidence=confidence,
                margin=margin,
            )
        return Routing(route="local", reason="confident", confidence=confidence, margin=margin)
