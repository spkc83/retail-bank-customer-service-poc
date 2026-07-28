"""Authenticated, model-driven synthetic retail-bank customer-service POC."""

# ZeroGPU must patch PyTorch before the CPU router imports it.
# ruff: noqa: I001

from __future__ import annotations

import html
import os
import uuid
from typing import Any

from zero_gpu_runtime import MODEL_REVISION, generate_final_answer, spaces_runtime

import gradio as gr

from auth import load_demo_auth
from model_service import GroundedBankingService, ModelResponseError
from orchestration import plan_workflow
from policy import (
    MODEL_FAILURE_RESPONSE,
    SENSITIVE_RESPONSE,
    contains_sensitive_value,
)
from router import ROUTER_REVISION, LearnedBankingRouter
from state import BANK

AUTH_CREDENTIALS = load_demo_auth()
SKIP_ROUTER_LOAD = os.environ.get("POC_SKIP_ROUTER_LOAD") == "1"
router = None if SKIP_ROUTER_LOAD else LearnedBankingRouter.from_hub()

CSS = """
.gradio-container { max-width: 1220px !important; }
.synthetic-banner {
  border: 1px solid #f0b429;
  background: #fff8df;
  border-radius: 12px;
  padding: 12px 16px;
}
.profile-card {
  border: 1px solid #dbe4f0;
  border-radius: 14px;
  padding: 14px;
  background: linear-gradient(145deg, #f8fbff, #eef5ff);
}
.status-ok { color: #087f5b; font-weight: 700; }
"""

PRESETS = [
    "Hello, how are you?",
    "Show my account balances.",
    "Show my five most recent transactions.",
    "Show transfers and recent transactions.",
    "What is the status of my debit card?",
    "My card was stolen. Freeze it.",
    "Please replace my debit card.",
    "I did not make the latest card purchase. Dispute it.",
    "Show my recent transfers.",
    "Cancel my pending transfer.",
    "Show my open service cases.",
    "When was my mailing address changed?",
    "Can you help me open a mortgage account?",
    "What is the weather tomorrow?",
]

PENDING_RESPONSE = (
    "The 9B model is preparing a grounded answer from the synthetic banking records…"
)

def respond(
    message: str,
    history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[str, str, str]:
    username, session_hash, plan, direct = _prepare_request(message, history, request)
    if direct is not None:
        return direct
    if plan is None:
        raise RuntimeError("model-backed request is missing a workflow plan")
    service = GroundedBankingService(
        bank=BANK,
        finalizer=generate_final_answer,
    )
    try:
        result = service.execute(
            username=username,
            session_hash=session_hash,
            message=message,
            history=history,
            plan=plan,
        )
    except ModelResponseError as error:
        return (
            _workflow_error_response(error),
            render_snapshot(BANK.snapshot(username, session_hash)),
            (
                "⚠️ The deterministic workflow or grounded model response failed "
                f"safely; no synthetic action was committed. Reason: {error}"
            ),
        )
    except (RuntimeError, ValueError):
        return (
            MODEL_FAILURE_RESPONSE,
            render_snapshot(BANK.snapshot(username, session_hash)),
            "⚠️ The model service was unavailable; no synthetic action was committed.",
        )
    workflow_label = " + ".join(result.workflow_tools)
    response_path = (
        "9B model finalizer"
        if result.selection_source == "model_finalizer"
        else "verified grounded repair after 9B validation"
    )
    response = (
        f"{result.response}\n\n"
        f"---\n_Model workflow: `{workflow_label.replace(' + ', '` + `')}` · "
        f"{response_path} · model revision `{MODEL_REVISION[:8]}…`_"
    )
    finalization_activity = (
        "the 9B model wrote the grounded final answer"
        if result.selection_source == "model_finalizer"
        else "server validation replaced an ungrounded model draft with verified results"
    )
    if plan.category == "single_write":
        activity = (
            f"✅ The deterministic workflow executed one explicit write "
            f"(`{workflow_label}`) inside the session transaction; "
            f"{finalization_activity} before commit."
        )
    else:
        activity = (
            f"✅ The deterministic workflow executed `{workflow_label}` against the "
            f"authenticated synthetic session, then {finalization_activity}."
        )
    return (
        response,
        render_snapshot(result.snapshot),
        activity,
    )


def dispatch_turn(
    message: str,
    history: list[dict[str, Any]],
    session_epoch: int,
    request: gr.Request,
) -> tuple[Any, list[dict[str, str]], list[dict[str, str]], str, str, Any, Any, Any, Any]:
    canonical_history = _canonical_history(history)
    username, session_hash, plan, direct = _prepare_request(
        message,
        canonical_history,
        request,
    )
    if direct is not None:
        response, snapshot, activity = direct
        completed = _with_assistant_turn(canonical_history, message, response)
        enabled = gr.update(interactive=True)
        return (
            gr.update(value="", interactive=True),
            completed,
            completed,
            snapshot,
            activity,
            gr.skip(),
            enabled,
            enabled,
            enabled,
        )
    if plan is None:
        raise RuntimeError("model-backed request is missing a workflow plan")
    pending = {
        "turn_id": uuid.uuid4().hex,
        "message": message.strip(),
        "history": canonical_history,
        "epoch": int(session_epoch),
    }
    visible = [
        *canonical_history,
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": PENDING_RESPONSE},
    ]
    disabled = gr.update(interactive=False)
    return (
        gr.update(value="", interactive=False),
        visible,
        canonical_history,
        render_snapshot(BANK.snapshot(username, session_hash)),
        (
            f"⏳ Workflow `{plan.category}` is waiting for the registered "
            "ZeroGPU model event."
        ),
        pending,
        disabled,
        disabled,
        disabled,
    )


@spaces_runtime.GPU(size="large", duration=90)
def finalize_turn(
    pending: dict[str, Any],
    session_epoch: int,
    current_history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, Any, Any, Any, Any]:
    message, history, pending_epoch = _pending_turn(pending)
    canonical_history = _canonical_history(current_history)
    if pending_epoch != int(session_epoch):
        username, session_hash = _identity(request)
        enabled = gr.update(interactive=True)
        return (
            canonical_history,
            canonical_history,
            render_snapshot(BANK.snapshot(username, session_hash)),
            "↺ The pending model turn expired because this demo session was reset.",
            gr.update(value="", interactive=True),
            enabled,
            enabled,
            enabled,
        )
    response, snapshot, activity = respond(message, history, request)
    completed = _with_assistant_turn(history, message, response)
    enabled = gr.update(interactive=True)
    return (
        completed,
        completed,
        snapshot,
        activity,
        gr.update(value="", interactive=True),
        enabled,
        enabled,
        enabled,
    )


def fail_pending_turn(
    pending: dict[str, Any],
    session_epoch: int,
    current_history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, Any, Any, Any, Any]:
    message, history, pending_epoch = _pending_turn(pending)
    canonical_history = _canonical_history(current_history)
    username, session_hash = _identity(request)
    enabled = gr.update(interactive=True)
    if pending_epoch != int(session_epoch):
        completed = canonical_history
        activity = "↺ The failed model turn was discarded after the demo session reset."
    else:
        completed = _with_assistant_turn(history, message, MODEL_FAILURE_RESPONSE)
        activity = (
            "⚠️ ZeroGPU could not allocate or complete the model turn. "
            "No synthetic write was committed."
        )
    return (
        completed,
        completed,
        render_snapshot(BANK.snapshot(username, session_hash)),
        activity,
        gr.update(value="", interactive=True),
        enabled,
        enabled,
        enabled,
    )


def route_query(
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if router is None:
        return {
            "route": "out_of_domain",
            "banking_probability": None,
            "intent": None,
            "threshold": None,
            "router_revision": ROUTER_REVISION,
            "reason": "router unavailable; failed closed",
        }
    try:
        return router.classify(message, history)
    except (TypeError, ValueError):
        return {
            "route": "out_of_domain",
            "banking_probability": None,
            "intent": None,
            "threshold": router.threshold,
            "router_revision": ROUTER_REVISION,
            "reason": "invalid route request; failed closed",
        }


def load_profile(request: gr.Request) -> tuple[str, str, str]:
    username, session_hash = _identity(request)
    snapshot = BANK.snapshot(username, session_hash)
    customer = snapshot["customer"]
    profile = (
        '<div class="profile-card">'
        f"<h3>{html.escape(str(customer['display_name']))}</h3>"
        f"<p><strong>{html.escape(str(customer['segment']))}</strong><br>"
        f"{html.escape(str(customer['city']))}<br>"
        f"Customer since {html.escape(str(customer['member_since']))}</p>"
        f"<p class='status-ok'>Authenticated as {html.escape(username)}</p>"
        "</div>"
    )
    return (
        profile,
        render_snapshot(snapshot),
        "Ready. Select a preset or ask the model to inspect the synthetic account.",
    )


def refresh_snapshot(request: gr.Request) -> str:
    username, session_hash = _identity(request)
    return render_snapshot(BANK.snapshot(username, session_hash))


def reset_demo(
    request: gr.Request,
) -> tuple[list[dict[str, str]], str, str]:
    username, session_hash = _identity(request)
    snapshot = BANK.reset(username, session_hash)
    return (
        [],
        render_snapshot(snapshot),
        "↺ This browser session was reset from the immutable synthetic seed.",
    )


def reset_session(
    session_epoch: int,
    request: gr.Request,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, int, Any, Any, Any, Any]:
    visible, snapshot, activity = reset_demo(request)
    enabled = gr.update(interactive=True)
    return (
        visible,
        visible,
        snapshot,
        activity,
        int(session_epoch) + 1,
        gr.update(value="", interactive=True),
        enabled,
        enabled,
        enabled,
    )


def render_snapshot(snapshot: dict[str, Any]) -> str:
    accounts = "\n".join(
        (
            f"- **{account['name']} ····{account['last4']}** — "
            f"{_money(account['available_balance_cents'], account['currency'])} available; "
            f"{_money(account['current_balance_cents'], account['currency'])} current"
        )
        for account in snapshot["accounts"]
    )
    cards = "\n".join(
        f"- **{card['name']} ····{card['last4']}** — `{card['status']}`"
        for card in snapshot["cards"]
    )
    transfers = "\n".join(
        (
            f"- {transfer['recipient']} — "
            f"{_money(transfer['amount_cents'], transfer['currency'])} "
            f"`{transfer['status']}`"
        )
        for transfer in snapshot["transfers"][:3]
    )
    cases = "\n".join(
        f"- {case['subject']} — `{case['status']}`"
        for case in snapshot["service_cases"][:4]
    )
    return (
        "### Synthetic backend state\n\n"
        f"#### Accounts\n{accounts or '- None'}\n\n"
        f"#### Cards\n{cards or '- None'}\n\n"
        f"#### Transfers\n{transfers or '- None'}\n\n"
        f"#### Service cases\n{cases or '- None'}"
    )


def _identity(request: gr.Request) -> tuple[str, str]:
    username = request.username
    session_hash = request.session_hash
    if not isinstance(username, str) or not username:
        raise ValueError("authenticated username is required")
    if not isinstance(session_hash, str) or not session_hash:
        raise ValueError("Gradio session hash is required")
    return username, session_hash


def _money(cents: Any, currency: Any) -> str:
    return f"{str(currency)} {int(cents) / 100:,.2f}"


def _workflow_error_response(error: ModelResponseError) -> str:
    reason = str(error).lower()
    if "not pending" in reason:
        return (
            "That synthetic transfer is already completed, so it cannot be cancelled. "
            "No synthetic data was changed."
        )
    return MODEL_FAILURE_RESPONSE


def _prepare_request(
    message: str,
    history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[str, str, Any, tuple[str, str, str] | None]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    username, session_hash = _identity(request)
    if contains_sensitive_value(message):
        return (
            username,
            session_hash,
            None,
            (
                SENSITIVE_RESPONSE,
                render_snapshot(BANK.snapshot(username, session_hash)),
                "🛡️ Credential guard blocked the request before routing or model inference.",
            ),
        )
    route = route_query(message, history)
    plan = plan_workflow(message, history, route)
    if plan.direct_response is None:
        return username, session_hash, plan, None
    return (
        username,
        session_hash,
        plan,
        (
            plan.direct_response,
            render_snapshot(BANK.snapshot(username, session_hash)),
            (
                f"🧭 No backend tool or ZeroGPU inference was needed. "
                f"Decision: `{plan.category}` ({plan.reason})."
            ),
        ),
    )


def _canonical_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    canonical: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        canonical.append({"role": str(role), "content": content})
    return canonical


def _with_assistant_turn(
    history: list[dict[str, str]],
    message: str,
    response: str,
) -> list[dict[str, str]]:
    return [
        *history,
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": response},
    ]


def _pending_turn(
    pending: dict[str, Any],
) -> tuple[str, list[dict[str, str]], int]:
    if not isinstance(pending, dict):
        raise ValueError("pending model turn must be an object")
    turn_id = pending.get("turn_id")
    message = pending.get("message")
    history = pending.get("history")
    epoch = pending.get("epoch")
    if not isinstance(turn_id, str) or not turn_id:
        raise ValueError("pending model turn is missing a turn ID")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("pending model turn is missing a message")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError("pending model turn has an invalid epoch")
    return message.strip(), _canonical_history(history), epoch


with gr.Blocks(
    title="Retail Bank Customer Service POC",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber"),
) as demo:
    gr.Markdown(
        """
        # Retail Bank Customer Service POC

        <div class="synthetic-banner">
        <strong>Fictional data only.</strong> The authenticated profiles, balances, cards,
        transactions, transfers, and actions are synthetic. The 9B model runs on ZeroGPU,
        writes grounded customer-facing answers after a deterministic workflow executes
        the supported backend operations. No real banking system is connected.
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            profile_panel = gr.HTML()
            snapshot_panel = gr.Markdown()
            activity_panel = gr.Markdown()
            with gr.Row():
                refresh_button = gr.Button("Refresh state", size="sm")
                reset_button = gr.Button("Reset demo", size="sm")
            gr.Button("Log out", link="/logout?all_session=false", size="sm")
        with gr.Column(scale=2, min_width=580):
            chatbot = gr.Chatbot(
                height=590,
                layout="bubble",
                type="messages",
                placeholder="Ask the model to inspect or update the synthetic bank profile.",
            )
            with gr.Row():
                message_box = gr.Textbox(
                    placeholder="Ask about the signed-in synthetic bank profile.",
                    show_label=False,
                    scale=8,
                    submit_btn=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(
                examples=[[prompt] for prompt in PRESETS],
                inputs=message_box,
                label="Preset test cases",
            )

    chat_history = gr.State([])
    pending_turn = gr.State(None)
    session_epoch = gr.State(0)
    route_message = gr.Textbox(visible=False)
    route_history = gr.JSON(visible=False)
    route_output = gr.JSON(visible=False)
    route_button = gr.Button(visible=False)
    route_button.click(
        route_query,
        inputs=[route_message, route_history],
        outputs=route_output,
        api_name="route",
        queue=False,
    )
    dispatch_event = gr.on(
        triggers=[message_box.submit, send_button.click],
        fn=dispatch_turn,
        inputs=[message_box, chat_history, session_epoch],
        outputs=[
            message_box,
            chatbot,
            chat_history,
            snapshot_panel,
            activity_panel,
            pending_turn,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name="chat",
        api_description=(
            "Authenticated CPU dispatch for synthetic retail-bank servicing chat."
        ),
        queue=True,
        trigger_mode="once",
    )
    model_event = pending_turn.change(
        finalize_turn,
        inputs=[pending_turn, session_epoch, chat_history],
        outputs=[
            chatbot,
            chat_history,
            snapshot_panel,
            activity_panel,
            message_box,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name=False,
        queue=True,
        trigger_mode="once",
    )
    model_event.failure(
        fail_pending_turn,
        inputs=[pending_turn, session_epoch, chat_history],
        outputs=[
            chatbot,
            chat_history,
            snapshot_panel,
            activity_panel,
            message_box,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name=False,
        queue=True,
    )
    demo.load(
        load_profile,
        outputs=[profile_panel, snapshot_panel, activity_panel],
        api_name=False,
    )
    refresh_button.click(
        refresh_snapshot,
        outputs=snapshot_panel,
        api_name="customer_snapshot",
        queue=False,
    )
    reset_button.click(
        reset_session,
        inputs=session_epoch,
        outputs=[
            chatbot,
            chat_history,
            snapshot_panel,
            activity_panel,
            session_epoch,
            message_box,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name="reset_demo",
        queue=False,
        cancels=[dispatch_event, model_event],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        auth=AUTH_CREDENTIALS,
        auth_message=(
            "<strong>Synthetic demo only.</strong> Use one of the two provided test accounts."
        ),
        ssr_mode=False,
        show_error=False,
    )
