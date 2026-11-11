"""蒸留データの生成元（シナリオ家族）。

1 つの家族 = 1 つの判断の型とドメイン。`semantic` のスロットが答えを決め、`decor` の
スロットは本文の見た目だけを変える（サービス名・人名・チケット番号など）。同じ家族・同じ
semantic でも、decor が違えば別の group として扱う。

答えはテンプレートの構成から決まる（LLM の推測ではない）。そのため `provenance.source` は
`synthetic`、`verified` は false になる。実教師で付け直す場合は `teacher/` のアダプタを使う。

テンプレートの `{xxx_text}` / `{xxx_text_ja}` は、`phrases` の同じ名前の表を
スロット `xxx` の値で引いて埋める。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SERVICES = (
    "checkout",
    "billing-api",
    "search",
    "auth",
    "notifications",
    "ingest",
    "reporting",
    "media",
    "webhook-dispatcher",
    "inventory",
)
REGIONS = ("us-east-1", "eu-west-1", "ap-northeast-1")
PEOPLE = ("A. Tanaka", "M. Silva", "J. Okafor", "L. Weber", "R. Iyer", "S. Novak")
TICKETS = tuple(str(number) for number in range(4001, 4121))
DATES = ("Monday", "Tuesday", "Wednesday", "Friday", "2026-09-30", "2026-10-15")
AMOUNTS = ("12.00", "19.99", "49.00", "120.00", "250.00")

CONTEXTS = (
    "the nightly batch",
    "the morning peak",
    "a canary rollout",
    "the migration window",
    "a load test",
    "the mobile client",
    "the admin console",
    "a retry storm",
)

DECOR = {
    "context": CONTEXTS,
    "service": SERVICES,
    "region": REGIONS,
    "person": PEOPLE,
    "ticket": TICKETS,
    "date": DATES,
    "amount": AMOUNTS,
}


@dataclass(frozen=True)
class Family:
    name: str
    domain: str
    qtype: str
    target: int
    semantic: dict[str, tuple[str, ...]]
    state: str
    label: Callable[[dict[str, str]], str]
    instructions: tuple[str, ...]
    criteria: dict[str, Any] | tuple[str, ...] | None = None
    state_ja: str | None = None
    instructions_ja: tuple[str, ...] | None = None
    criteria_ja: dict[str, Any] | tuple[str, ...] | None = None
    phrases: dict[str, dict[str, str]] = field(default_factory=dict)
    decor: dict[str, tuple[str, ...]] = field(default_factory=lambda: DECOR)
    japanese_ratio: float = 0.6
    reserved: str | None = None


# --- Choice -----------------------------------------------------------------

OPS_RATE_LIMIT = Family(
    name="ops_rate_limit",
    domain="software_operations",
    qtype="choice",
    target=350,
    semantic={"delay": ("30", "120", "300")},
    state="{service} in {region} answered HTTP 429 during {context}. Retry-After: {delay}.",
    instructions=("Choose the next safe action.", "What should the client do next?"),
    criteria={
        "retry_30s": "Wait 30 seconds",
        "retry_2m": "Wait 2 minutes",
        "retry_5m": "Wait 5 minutes",
        "abort": "Stop the task",
    },
    label=lambda slots: {"30": "retry_30s", "120": "retry_2m", "300": "retry_5m"}[slots["delay"]],
    state_ja="{service}（{region}）が HTTP 429 を返し、Retry-After: {delay} が付いています。",
    instructions_ja=("次に取るべき行動はどれですか。",),
    criteria_ja={
        "retry_30s": "30 秒待つ",
        "retry_2m": "2 分待つ",
        "retry_5m": "5 分待つ",
        "abort": "タスクを中止する",
    },
)

OPS_DISK = Family(
    name="ops_disk",
    domain="software_operations",
    qtype="choice",
    target=250,
    semantic={"usage": ("91", "95", "98"), "growth": ("1", "3", "6")},
    state="{service} in {region} is at {usage}% disk usage. Logs grow by {growth} GB per hour.",
    instructions=("What should be done first?", "Which action keeps the service running?"),
    criteria={
        "rotate_logs": "Rotate or truncate the logs",
        "ignore": "Ignore it, some space is still free",
        "restart": "Restart the service",
    },
    label=lambda slots: "rotate_logs",
    state_ja="{service}（{region}）のディスク使用率が {usage}% で、ログが 1 時間に {growth} GB 増えています。",
    instructions_ja=("最初にやるべきことはどれですか。",),
    criteria_ja={
        "rotate_logs": "ログをローテートして空きを作る",
        "ignore": "まだ空きがあるので放置する",
        "restart": "サービスを再起動する",
    },
)

DEPLOY_FAILURE = Family(
    name="deploy_failure",
    domain="software_operations",
    qtype="choice",
    target=220,
    semantic={"cause": ("image_missing", "migration_failed", "out_of_memory")},
    state="Deploy of {service} to {region} on {date} failed: {cause_text}.",
    instructions=("What is the first step?", "Which action matches the failure?"),
    criteria={
        "check_tag": "Confirm the image tag exists and CI pushed it",
        "rollback": "Roll back the previous release",
        "scale_up": "Add more replicas",
    },
    label=lambda slots: {
        "image_missing": "check_tag",
        "migration_failed": "rollback",
        "out_of_memory": "scale_up",
    }[slots["cause"]],
    phrases={
        "cause_text": {
            "image_missing": "the image tag sha-9f3c was not found in the registry",
            "migration_failed": "the database migration failed halfway",
            "out_of_memory": "the container was killed after running out of memory",
        }
    },
)

BILLING_DUPLICATE = Family(
    name="billing_duplicate",
    domain="billing",
    qtype="choice",
    target=220,
    semantic={"count": ("2", "3")},
    state="{person} was charged {count} times for the same order on {date}, {amount} each.",
    instructions=("What should be done?", "Which remedy fits?"),
    criteria={
        "refund_duplicate": "Refund the duplicate charges",
        "explain": "Explain that repeated charges are expected",
        "charge_next": "Charge the next month early",
    },
    label=lambda slots: "refund_duplicate",
    state_ja="{person} さんが {date} に同じ注文で {count} 回、{amount} ずつ請求されています。",
    instructions_ja=("正しい対応はどれですか。",),
    criteria_ja={
        "refund_duplicate": "重複した分を返金する",
        "explain": "重複請求は仕様だと説明する",
        "charge_next": "翌月分を前倒しで請求する",
    },
)

TICKET_ROUTING = Family(
    name="ticket_routing",
    domain="customer_support",
    qtype="choice",
    target=320,
    semantic={"topic": ("refund", "login", "shipping", "invoice")},
    state="Ticket {ticket}: {topic_text}.",
    instructions=("Which team should handle this?", "Who owns this ticket?"),
    criteria={
        "billing": "Refunds, charges, and invoices",
        "account_access": "Login, password, and account access",
        "logistics": "Delivery status and shipments",
        "finance": "Invoice corrections and payment terms",
    },
    label=lambda slots: {
        "refund": "billing",
        "login": "account_access",
        "shipping": "logistics",
        "invoice": "finance",
    }[slots["topic"]],
    phrases={
        "topic_text": {
            "refund": "the customer asks for a refund for order {ticket}",
            "login": "the customer cannot sign in after a password reset",
            "shipping": "the customer asks where the parcel is",
            "invoice": "the customer says the invoice total does not match the lines",
        }
    },
)

SECURITY_TOKEN = Family(
    name="security_token",
    domain="security_incidents",
    qtype="choice",
    target=180,
    semantic={"exposure": ("public_repo", "chat_channel", "screenshot")},
    state="An access token for {service} appeared in {exposure_text} (reported by {person}) about 20 minutes ago.",
    instructions=("What must happen first?", "Which step comes first?"),
    criteria={
        "revoke_token": "Revoke the token immediately",
        "ignore": "Ignore it, the message can be deleted later",
        "change_password": "Only change the account password",
    },
    label=lambda slots: "revoke_token",
    state_ja="アクセストークンが{exposure_text_ja}に出てから 20 分ほど経っています（{service}、{person} さんの報告）。",
    instructions_ja=("最優先で行うことはどれですか。",),
    criteria_ja={
        "revoke_token": "トークンを無効化する",
        "ignore": "あとで消せばよいので放置する",
        "change_password": "パスワードだけ変更する",
    },
    phrases={
        "exposure_text": {
            "public_repo": "a public repository",
            "chat_channel": "a shared chat channel",
            "screenshot": "a screenshot attached to a ticket",
        },
        "exposure_text_ja": {
            "public_repo": "公開リポジトリ",
            "chat_channel": "共有チャット",
            "screenshot": "チケットのスクリーンショット",
        },
    },
)

OPS_CONFIG_CHANGE = Family(
    name="ops_config_change",
    domain="software_operations",
    qtype="choice",
    target=120,
    semantic={"symptom": ("latency", "errors", "throughput")},
    state="Right after a configuration change on {service} in {region} on {date}, {symptom_text}.",
    instructions=("What is the best next action?", "What should the operator do?"),
    criteria={
        "revert_change": "Revert the change and measure again",
        "add_replicas": "Add replicas without checking",
        "wait": "Wait and change nothing",
    },
    label=lambda slots: "revert_change",
    phrases={
        "symptom_text": {
            "latency": "the p95 latency doubled",
            "errors": "the error rate rose to 30%",
            "throughput": "the throughput dropped by half",
        }
    },
)

LOGISTICS_ADDRESS = Family(
    name="logistics_address",
    domain="logistics",
    qtype="choice",
    target=120,
    reserved="unseen_domain",
    semantic={"status": ("not_shipped", "shipped", "delivered")},
    state="Order {ticket} for {person}: the customer asks to change the delivery address. The parcel is {status_text}.",
    instructions=("Which option matches the situation?", "What is the correct action?"),
    criteria={
        "change_address": "Change the address before dispatch",
        "contact_carrier": "Ask the carrier to redirect the parcel",
        "refund": "Refund the order and let the customer reorder",
    },
    label=lambda slots: {
        "not_shipped": "change_address",
        "shipped": "contact_carrier",
        "delivered": "refund",
    }[slots["status"]],
    phrases={
        "status_text": {
            "not_shipped": "not shipped yet",
            "shipped": "already in transit",
            "delivered": "marked as delivered yesterday",
        }
    },
)

INSURANCE_COVERAGE = Family(
    name="insurance_coverage",
    domain="insurance_claims",
    qtype="choice",
    target=100,
    reserved="unseen_domain",
    semantic={"period": ("active", "expired"), "damage": ("hail", "collision", "water")},
    state="A claim for {amount} reports {damage} damage on {date}. The policy {period_text} on that date.",
    instructions=("What should the adjuster do?", "Which decision fits the policy?"),
    criteria={
        "pay_claim": "Pay the claim",
        "deny_claim": "Deny the claim as outside the policy period",
        "inspect": "Order an inspection first",
    },
    label=lambda slots: {"active": "pay_claim", "expired": "deny_claim"}[slots["period"]],
    phrases={"period_text": {"active": "was active", "expired": "had expired"}},
)

EDUCATION_LATE = Family(
    name="education_late",
    domain="education",
    qtype="choice",
    target=80,
    reserved="unseen_domain",
    semantic={"evidence": ("documented", "none")},
    state="A student (ref {ticket}) submitted the assignment 5 hours late on {date} and {evidence_text}.",
    instructions=("What fits the usual policy?", "Which action follows the late policy?"),
    criteria={
        "review_evidence": "Review the evidence against the late policy",
        "auto_fail": "Fail the assignment automatically",
        "ignore": "Ignore the submission",
    },
    label=lambda slots: "review_evidence",
    phrases={
        "evidence_text": {
            "documented": "submitted a medical certificate",
            "none": "gave no evidence at all",
        }
    },
)

SUPPORT_CRITERIA = Family(
    name="support_criteria_variant",
    domain="customer_support",
    qtype="choice",
    target=120,
    reserved="unseen_criteria",
    semantic={"topic": ("refund", "login", "shipping", "invoice")},
    state="Ticket {ticket} from {person}: {topic_text}.",
    instructions=("Choose the team that should own this ticket.",),
    criteria={
        "team_money": "Money: refunds, chargebacks, and duplicate charges",
        "team_access": "Access: sign-in, passwords, and locked accounts",
        "team_delivery": "Delivery: parcels, delays, and lost shipments",
        "team_documents": "Documents: invoices, receipts, and tax forms",
        "team_other": "Anything that does not fit the four teams above",
    },
    label=lambda slots: {
        "refund": "team_money",
        "login": "team_access",
        "shipping": "team_delivery",
        "invoice": "team_documents",
    }[slots["topic"]],
    phrases={
        "topic_text": {
            "refund": "asks for a refund of order {ticket}",
            "login": "cannot sign in after a password reset",
            "shipping": "asks where the parcel is",
            "invoice": "says the invoice total does not match the lines",
        }
    },
)

OPS_QUESTION_VARIANT = Family(
    name="ops_question_variant",
    domain="software_operations",
    qtype="choice",
    target=120,
    reserved="unseen_question",
    semantic={"delay": ("30", "300")},
    state=(
        "The upstream service {service} rejected the call with 429 during {context} "
        "and asked us to wait {delay} seconds."
    ),
    instructions=("Which retry policy follows the instruction from the server?",),
    criteria={
        "backoff_jitter": "Back off with jitter for the requested delay",
        "retry_now": "Retry now in a tight loop",
        "give_up": "Give up permanently",
    },
    label=lambda slots: "backoff_jitter",
)


# --- Noul -------------------------------------------------------------------

REFUND_REQUEST = Family(
    name="refund_request",
    domain="billing",
    qtype="noul",
    target=250,
    semantic={"intent": ("refund", "exchange", "invoice_question")},
    state="{person}: {intent_text}.",
    instructions=("Is the customer asking for money back?", "Does the message request a refund?"),
    label=lambda slots: "yes" if slots["intent"] == "refund" else "no",
    state_ja="{person} さん: {intent_text_ja}",
    instructions_ja=("返金を求めていますか。",),
    phrases={
        "intent_text": {
            "refund": "Please return the money I paid for order {ticket}",
            "exchange": "Can I swap the size of order {ticket}?",
            "invoice_question": "Where can I download the invoice for order {ticket}?",
        },
        "intent_text_ja": {
            "refund": "注文 {ticket} の代金を返してください。",
            "exchange": "注文 {ticket} のサイズを交換できますか。",
            "invoice_question": "注文 {ticket} の請求書はどこから取得できますか。",
        },
    },
)

HUMAN_ESCALATION = Family(
    name="human_escalation",
    domain="customer_support",
    qtype="noul",
    target=250,
    semantic={"intent": ("human", "thanks", "howto", "repeat")},
    state="Ticket {ticket} from {person}: {intent_text}",
    instructions=(
        "Is the customer asking for a human agent?",
        "Does the customer want a person to handle this?",
    ),
    label=lambda slots: "yes" if slots["intent"] in {"human", "repeat"} else "no",
    state_ja="チケット {ticket}（{person} さん）: {intent_text_ja}",
    instructions_ja=("人間の担当者を求めていますか。",),
    phrases={
        "intent_text": {
            "human": "Please connect me to a real person.",
            "thanks": "Thanks, that fixed it. Nothing else needed.",
            "howto": "How do I change my password?",
            "repeat": "This is my third message about the same problem. I want a person who can decide.",
        },
        "intent_text_ja": {
            "human": "人間の担当者につないでください。",
            "thanks": "ありがとうございます。解決しました。",
            "howto": "パスワードの変更方法を教えてください。",
            "repeat": "同じ件で三度目の連絡です。判断できる人に対応してほしいです。",
        },
    },
)

CREDENTIAL_LEAK = Family(
    name="credential_leak",
    domain="security_incidents",
    qtype="noul",
    target=150,
    semantic={"exposure": ("secret", "forgot")},
    state="Ticket {ticket} from {person}: {exposure_text}",
    instructions=("Does the message contain a credential?", "Is a secret visible in the message?"),
    label=lambda slots: "yes" if slots["exposure"] == "secret" else "no",
    state_ja="チケット {ticket}（{person} さん）: {exposure_text_ja}",
    instructions_ja=("本文に認証情報が含まれていますか。",),
    phrases={
        "exposure_text": {
            "secret": "My password is hunter2, please check why the login fails.",
            "forgot": "I forgot my password, please send a reset link.",
        },
        "exposure_text_ja": {
            "secret": "パスワードは hunter2 です。ログインできない原因を調べてください。",
            "forgot": "パスワードを忘れました。再設定のリンクを送ってください。",
        },
    },
)

DUPLICATE_REPORT = Family(
    name="duplicate_report",
    domain="customer_support",
    qtype="noul",
    target=150,
    semantic={"repeat": ("first", "third")},
    state="Ticket {ticket}: {repeat_text} that {service} fails on upload.",
    instructions=("Is this a duplicate report?", "Has the customer reported this before?"),
    label=lambda slots: "yes" if slots["repeat"] == "third" else "no",
    phrases={
        "repeat_text": {
            "first": "This is the first report",
            "third": "This is the third time I report",
        }
    },
)

ROLLBACK_SAFE = Family(
    name="rollback_safe",
    domain="software_operations",
    qtype="noul",
    target=200,
    semantic={"plan": ("tested_safe", "untested_delete")},
    state="Migration on {service} ({date}): {plan_text}.",
    instructions=("Is a rollback safe to perform now?", "Can the change be undone safely?"),
    label=lambda slots: "yes" if slots["plan"] == "tested_safe" else "no",
    phrases={
        "plan_text": {
            "tested_safe": "the rollback script was tested and no rows are deleted",
            "untested_delete": "the rollback script has never run and the migration deletes rows",
        }
    },
)

SHIPMENT_DELAYED = Family(
    name="shipment_delayed",
    domain="logistics",
    qtype="noul",
    target=150,
    reserved="unseen_domain",
    semantic={"status": ("stuck", "ontime")},
    state="Parcel {ticket} for {person}: {status_text}.",
    instructions=("Is the shipment delayed?",),
    label=lambda slots: "yes" if slots["status"] == "stuck" else "no",
    phrases={
        "status_text": {
            "stuck": "it has not moved for six days",
            "ontime": "it arrived on time",
        }
    },
)

MEDICAL_EMERGENCY = Family(
    name="medical_emergency",
    domain="healthcare_support",
    qtype="noul",
    target=120,
    reserved="unseen_domain",
    semantic={"symptom": ("emergency", "mild")},
    state="{person} writes on {date} (patient ref {ticket}): {symptom_text}",
    instructions=("Does this message describe a possible emergency?",),
    label=lambda slots: "yes" if slots["symptom"] == "emergency" else "no",
    phrases={
        "symptom_text": {
            "emergency": "I have chest pain that spreads to my arm and I cannot breathe.",
            "mild": "I have a mild rash after changing soap.",
        }
    },
)

NEGATION = Family(
    name="negation",
    domain="software_operations",
    qtype="noul",
    target=180,
    reserved="adversarial",
    semantic={"case": ("deploy_positive", "deploy_negated", "invoice_positive", "invoice_negated")},
    state="Note from {person} on {date}: {case_text}",
    instructions=("Is the proposition true?", "Does the statement hold?"),
    label=lambda slots: "no" if slots["case"].endswith("_negated") else "yes",
    phrases={
        "case_text": {
            "deploy_positive": "The deploy failed and the rollback is running.",
            "deploy_negated": "It is not the case that the deploy failed. All health checks passed.",
            "invoice_positive": "The invoice is unpaid and the due date has passed.",
            "invoice_negated": "It is not the case that the invoice is unpaid. It was paid last week.",
        }
    },
)

INFO_SUFFICIENT = Family(
    name="info_sufficient",
    domain="customer_support",
    qtype="noul",
    target=200,
    reserved="adversarial",
    semantic={"detail": ("empty", "vague", "detailed")},
    state="Ticket {ticket} from {person}: {detail_text}",
    instructions=("Is there enough information to decide what the customer wants?",),
    label=lambda slots: "no" if slots["detail"] in {"empty", "vague"} else "yes",
    phrases={
        "detail_text": {
            "empty": "...",
            "vague": "something is broken",
            "detailed": "the export button crashes on Safari 17 when the settings page loads, Chrome works",
        }
    },
)


# --- Score ------------------------------------------------------------------

SEVERITY = Family(
    name="severity",
    domain="software_operations",
    qtype="score",
    target=350,
    semantic={"symptom": ("cosmetic", "workaround", "blocking")},
    state="Report on {service} ({date}): {symptom_text}.",
    instructions=("How severe is the reported issue?", "Rate the impact of this defect."),
    criteria=(
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but a workaround exists",
        "Blocking issue; no workaround exists",
    ),
    label=lambda slots: {"cosmetic": "0", "workaround": "1", "blocking": "2"}[slots["symptom"]],
    state_ja="{service}（{date}）の報告: {symptom_text_ja}",
    instructions_ja=("この不具合の深刻度はどれですか。",),
    criteria_ja=(
        "見た目だけで機能に影響しない",
        "機能は劣化するが回避策がある",
        "回避策がなく作業が止まる",
    ),
    phrases={
        "symptom_text": {
            "cosmetic": "a label is misaligned by two pixels",
            "workaround": "upload fails in one browser but works in another",
            "blocking": "nobody can sign in and every attempt returns 500",
        },
        "symptom_text_ja": {
            "cosmetic": "ラベルが 2 ピクセルずれている",
            "workaround": "一部のブラウザでアップロードが失敗するが他は動く",
            "blocking": "全員がログインできず毎回 500 が返る",
        },
    },
)

URGENCY = Family(
    name="urgency",
    domain="customer_support",
    qtype="score",
    target=300,
    semantic={"deadline": ("none", "soon", "now")},
    state="{person} (ticket {ticket}): {deadline_text}",
    instructions=("How urgent is this request?", "How time-critical is the request?"),
    criteria=(
        "No deadline; a general question",
        "A deadline exists but time remains",
        "Blocking now; money or access is lost today",
    ),
    label=lambda slots: {"none": "0", "soon": "1", "now": "2"}[slots["deadline"]],
    state_ja="{person} さん（チケット {ticket}）: {deadline_text_ja}",
    instructions_ja=("この依頼の緊急度はどれですか。",),
    criteria_ja=("期限なし。一般的な質問", "期限はあるが余裕がある", "今日中に止まってしまう"),
    phrases={
        "deadline_text": {
            "none": "how much does the business plan cost? No rush.",
            "soon": "I need the invoice by Friday and today is Tuesday.",
            "now": "payment fails for every customer and checkout is down right now.",
        },
        "deadline_text_ja": {
            "none": "料金プランについて一般的な質問です。急ぎません。",
            "soon": "金曜までに請求書が必要で、今日は火曜です。",
            "now": "すべての決済が失敗していて、今この瞬間に売上が落ちています。",
        },
    },
)

FRUSTRATION = Family(
    name="frustration",
    domain="customer_support",
    qtype="score",
    target=250,
    semantic={"tone": ("calm", "civil", "angry")},
    state="Customer {person} writes about {service} on {date}: {tone_text}",
    instructions=("How frustrated is the customer?",),
    criteria=("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threats"),
    label=lambda slots: {"calm": "0", "civil": "1", "angry": "2"}[slots["tone"]],
    state_ja="お客様 {person} さんから {service} について {date} の連絡: {tone_text_ja}",
    instructions_ja=("利用者のいらだちはどの程度ですか。",),
    criteria_ja=("落ち着いて事実を述べている", "不満はあるが礼儀は保っている", "強い言葉や脅しがある"),
    phrases={
        "tone_text": {
            "calm": "The report is clear and factual: the export fails on Safari.",
            "civil": "I have waited a week with no answer. This is not acceptable.",
            "angry": "This is the third time and nobody helps. I will cancel everything.",
        },
        "tone_text_ja": {
            "calm": "落ち着いて事実を書きます。Safari でエクスポートが失敗します。",
            "civil": "1 週間返事がなく、これでは困ります。",
            "angry": "三度目です。誰も対応してくれないので、契約を全部やめます。",
        },
    },
)

REPORT_QUALITY = Family(
    name="report_quality",
    domain="customer_support",
    qtype="score",
    target=100,
    semantic={"detail": ("none", "feature_only", "partial", "full")},
    state="Bug report on {service} ({date}): {detail_text}",
    instructions=("How much does the report give an engineer to work with?",),
    criteria=(
        "No detail; just says something is broken",
        "Names the feature but no steps or environment",
        "Steps to reproduce or environment, but not both",
        "Steps to reproduce and environment",
    ),
    label=lambda slots: {"none": "0", "feature_only": "1", "partial": "2", "full": "3"}[slots["detail"]],
    phrases={
        "detail_text": {
            "none": "something is broken.",
            "feature_only": "the export feature is broken.",
            "partial": "the export button crashes on Safari 17.",
            "full": "the export button crashes on Safari 17 when the settings page loads; Chrome works.",
        }
    },
)

EVIDENCE_NEEDED = Family(
    name="evidence_needed",
    domain="insurance_claims",
    qtype="score",
    target=100,
    reserved="unseen_domain",
    semantic={"case": ("minor", "contested", "fraud")},
    state="Claim {ticket} for {amount} filed on {date}: {case_text}.",
    instructions=("How much evidence is needed?",),
    criteria=("Photos are enough", "An inspection is needed", "A full investigation is needed"),
    label=lambda slots: {"minor": "0", "contested": "1", "fraud": "2"}[slots["case"]],
    phrases={
        "case_text": {
            "minor": "a scratched bumper with photos attached",
            "contested": "two drivers disagree about who hit whom",
            "fraud": "the same damage was claimed three times under different policies",
        }
    },
)


FAMILIES: tuple[Family, ...] = (
    OPS_RATE_LIMIT,
    OPS_DISK,
    DEPLOY_FAILURE,
    BILLING_DUPLICATE,
    TICKET_ROUTING,
    SECURITY_TOKEN,
    OPS_CONFIG_CHANGE,
    LOGISTICS_ADDRESS,
    INSURANCE_COVERAGE,
    EDUCATION_LATE,
    SUPPORT_CRITERIA,
    OPS_QUESTION_VARIANT,
    REFUND_REQUEST,
    HUMAN_ESCALATION,
    CREDENTIAL_LEAK,
    DUPLICATE_REPORT,
    ROLLBACK_SAFE,
    SHIPMENT_DELAYED,
    MEDICAL_EMERGENCY,
    NEGATION,
    INFO_SUFFICIENT,
    SEVERITY,
    URGENCY,
    FRUSTRATION,
    REPORT_QUALITY,
    EVIDENCE_NEEDED,
)
