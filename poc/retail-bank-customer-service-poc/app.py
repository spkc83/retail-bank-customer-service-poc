"""Authenticated model-driven synthetic retail-bank conversational POC."""

# ZeroGPU must patch PyTorch before the CPU router imports it.
# ruff: noqa: I001

from __future__ import annotations

import html
import json
import os
import uuid
from typing import Any

if os.environ.get("POC_SKIP_MODEL_LOAD") == "1":
    from zero_gpu_runtime import (
        MODEL_REVISION,
        count_tokens,
        generate_text,
        spaces_runtime as spaces,
    )
else:
    import spaces

    from zero_gpu_runtime import MODEL_REVISION, count_tokens, generate_text

import gradio as gr

from auth import load_demo_auth
from model_service import (
    AgentExecutionError,
    AgentProtocolError,
    ConversationalBankingAgent,
    ModelRuntime,
    ToolCall,
    canonical_conversation,
)
from policy import (
    MODEL_FAILURE_RESPONSE,
    OOD_RESPONSE,
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
    "yo, sup?",
    "Show my account balances.",
    "What happened with the money I sent recently?",
    "Show my five most recent transactions.",
    "What is the status of my debit card?",
    "My card was stolen. Freeze it.",
    "Please replace my debit card.",
    "I did not make the North Harbor Market purchase. Dispute it.",
    "Cancel the pending transfer to River Consulting.",
    "When was my mailing address changed?",
    "Can you help me open a mortgage account?",
    "What is the weather tomorrow?",
]

PENDING_RESPONSE = "The 9B model is thinking and may call the synthetic bank tools…"


class _RuntimeModel(ModelRuntime):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> str:
        return generate_text(messages, tools, max_new_tokens)

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        return count_tokens(messages, tools)


def dispatch_turn(
    message: str,
    visible_history: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    session_epoch: int,
    request: gr.Request,
) -> tuple[Any, list[dict[str, str]], list[dict[str, Any]], str, str, str, Any, Any, Any, Any]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    username, session_hash = _identity(request)
    visible = _visible_history(visible_history)
    conversation = canonical_conversation(conversation_history)

    if contains_sensitive_value(message):
        return _direct_turn(
            message=message,
            response=SENSITIVE_RESPONSE,
            visible=visible,
            conversation=conversation,
            snapshot=render_snapshot(BANK.snapshot(username, session_hash)),
            activity="Credential-value input was rejected before routing or model inference.",
            diagnostics="### Experiment diagnostics\n\nInput guard: credential value detected.",
        )

    route = route_query(message, conversation)
    if route.get("route") == "out_of_domain":
        return _direct_turn(
            message=message,
            response=OOD_RESPONSE,
            visible=visible,
            conversation=conversation,
            snapshot=render_snapshot(BANK.snapshot(username, session_hash)),
            activity="High-confidence OOD head decision; the 9B model was not invoked.",
            diagnostics=_render_diagnostics(route, (), (), "OOD stock response"),
        )

    pending = {
        "turn_id": uuid.uuid4().hex,
        "message": message.strip(),
        "conversation": conversation,
        "router_result": route,
        "epoch": int(session_epoch),
    }
    pending_visible = [
        *visible,
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": PENDING_RESPONSE},
    ]
    disabled = gr.update(interactive=False)
    return (
        gr.update(value="", interactive=False),
        pending_visible,
        conversation,
        render_snapshot(BANK.snapshot(username, session_hash)),
        "The allowed/uncertain turn is queued for the registered 9B ZeroGPU event.",
        _render_diagnostics(route, (), (), "waiting for 9B model"),
        pending,
        disabled,
        disabled,
        disabled,
    )


@spaces.GPU(size="large", duration=90)
def finalize_turn(
    pending: dict[str, Any],
    session_epoch: int,
    current_visible: list[dict[str, Any]],
    current_conversation: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str, str, str, Any, Any, Any, Any]:
    message, conversation, router_result, pending_epoch = _pending_turn(pending)
    if pending_epoch != int(session_epoch):
        username, session_hash = _identity(request)
        enabled = gr.update(interactive=True)
        return (
            _visible_history(current_visible),
            canonical_conversation(current_conversation),
            render_snapshot(BANK.snapshot(username, session_hash)),
            "The queued model turn expired because this demo session was reset.",
            "### Experiment diagnostics\n\nStale turn discarded.",
            gr.update(value="", interactive=True),
            enabled,
            enabled,
            enabled,
        )

    username, session_hash = _identity(request)
    agent = ConversationalBankingAgent(bank=BANK, model=_RuntimeModel())
    try:
        result = agent.run_turn(
            username=username,
            session_hash=session_hash,
            message=message,
            conversation=conversation,
            router_result=router_result,
        )
    except AgentExecutionError as error:
        failed_conversation = [
            *error.conversation,
            {"role": "assistant", "content": MODEL_FAILURE_RESPONSE},
        ]
        enabled = gr.update(interactive=True)
        return (
            _visible_from_conversation(failed_conversation),
            failed_conversation,
            render_snapshot(error.snapshot),
            (
                "The 9B model failed after executing the tool calls shown in "
                "diagnostics. No CPU-authored servicing answer was substituted."
            ),
            (
                f"{_render_diagnostics(
                    router_result,
                    error.tool_calls,
                    error.tool_results,
                    '9B second-pass failure',
                )}\n\n"
                f"Failure type: `{type(error.__cause__).__name__}`"
            ),
            gr.update(value="", interactive=True),
            enabled,
            enabled,
            enabled,
        )
    except (AgentProtocolError, RuntimeError, TypeError, ValueError) as error:
        failed_conversation = _with_text_turn(
            conversation,
            message,
            MODEL_FAILURE_RESPONSE,
        )
        enabled = gr.update(interactive=True)
        return (
            _visible_from_conversation(failed_conversation),
            failed_conversation,
            render_snapshot(BANK.snapshot(username, session_hash)),
            (
                "The 9B model event failed. No CPU-authored servicing answer was "
                "substituted; the synthetic dashboard shows the current backend state."
            ),
            (
                f"{_render_diagnostics(router_result, (), (), '9B model failure')}\n\n"
                f"Failure type: `{type(error).__name__}`"
            ),
            gr.update(value="", interactive=True),
            enabled,
            enabled,
            enabled,
        )

    enabled = gr.update(interactive=True)
    return (
        _visible_from_conversation(result.conversation),
        result.conversation,
        render_snapshot(result.snapshot),
        (
            "The 9B model authored the response directly."
            if not result.tool_calls
            else (
                "The 9B model selected and called the synthetic tools, then authored "
                "the final response from their results."
            )
        ),
        _render_diagnostics(
            router_result,
            result.tool_calls,
            result.tool_results,
            "9B model-authored",
        ),
        gr.update(value="", interactive=True),
        enabled,
        enabled,
        enabled,
    )


def fail_pending_turn(
    pending: dict[str, Any],
    session_epoch: int,
    current_visible: list[dict[str, Any]],
    current_conversation: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str, str, str, Any, Any, Any, Any]:
    message, conversation, router_result, pending_epoch = _pending_turn(pending)
    username, session_hash = _identity(request)
    enabled = gr.update(interactive=True)
    if pending_epoch != int(session_epoch):
        return (
            _visible_history(current_visible),
            canonical_conversation(current_conversation),
            render_snapshot(BANK.snapshot(username, session_hash)),
            "The failed model turn was discarded after the demo session reset.",
            "### Experiment diagnostics\n\nStale failed turn discarded.",
            gr.update(value="", interactive=True),
            enabled,
            enabled,
            enabled,
        )
    failed_conversation = _with_text_turn(
        conversation,
        message,
        MODEL_FAILURE_RESPONSE,
    )
    return (
        _visible_from_conversation(failed_conversation),
        failed_conversation,
        render_snapshot(BANK.snapshot(username, session_hash)),
        (
            "ZeroGPU could not allocate or complete the 9B model turn. "
            "No CPU-authored servicing answer was substituted."
        ),
        _render_diagnostics(router_result, (), (), "ZeroGPU/model unavailable"),
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
        return _uncertain_route("router unavailable; delegated to the 9B model")
    try:
        return router.classify(message, history)
    except (RuntimeError, TypeError, ValueError):
        return _uncertain_route("router failed; delegated to the 9B model")


def load_profile(request: gr.Request) -> tuple[str, str, str, str]:
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
        "Ready. Allowed turns are handled by the 9B model; high-confidence OOD is gated.",
        "### Experiment diagnostics\n\nNo turn has run yet.",
    )


def refresh_snapshot(request: gr.Request) -> str:
    username, session_hash = _identity(request)
    return render_snapshot(BANK.snapshot(username, session_hash))


def reset_session(
    session_epoch: int,
    request: gr.Request,
) -> tuple[list[Any], list[Any], str, str, int, Any, Any, Any, Any]:
    username, session_hash = _identity(request)
    snapshot = BANK.reset(username, session_hash)
    enabled = gr.update(interactive=True)
    return (
        [],
        [],
        render_snapshot(snapshot),
        "This browser session and complete model conversation were reset.",
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


def _direct_turn(
    *,
    message: str,
    response: str,
    visible: list[dict[str, str]],
    conversation: list[dict[str, Any]],
    snapshot: str,
    activity: str,
    diagnostics: str,
) -> tuple[Any, list[dict[str, str]], list[dict[str, Any]], str, str, str, Any, Any, Any, Any]:
    completed_conversation = _with_text_turn(conversation, message, response)
    completed_visible = [
        *visible,
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": response},
    ]
    enabled = gr.update(interactive=True)
    return (
        gr.update(value="", interactive=True),
        completed_visible,
        completed_conversation,
        snapshot,
        activity,
        diagnostics,
        gr.skip(),
        enabled,
        enabled,
        enabled,
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


def _visible_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and str(item["content"]).strip()
    ]


def _visible_from_conversation(
    conversation: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return _visible_history(conversation)


def _with_text_turn(
    conversation: list[dict[str, Any]],
    message: str,
    response: str,
) -> list[dict[str, Any]]:
    return [
        *canonical_conversation(conversation),
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": response},
    ]


def _pending_turn(
    pending: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any], int]:
    if not isinstance(pending, dict):
        raise ValueError("pending model turn must be an object")
    if not isinstance(pending.get("turn_id"), str) or not pending["turn_id"]:
        raise ValueError("pending model turn is missing a turn ID")
    message = pending.get("message")
    router_result = pending.get("router_result")
    epoch = pending.get("epoch")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("pending model turn is missing a message")
    if not isinstance(router_result, dict):
        raise ValueError("pending model turn is missing router diagnostics")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError("pending model turn has an invalid epoch")
    return (
        message.strip(),
        canonical_conversation(pending.get("conversation")),
        router_result,
        epoch,
    )


def _uncertain_route(reason: str) -> dict[str, Any]:
    return {
        "route": "uncertain",
        "banking_probability": None,
        "ood_probability": None,
        "confidence": None,
        "intent": None,
        "intent_confidence": None,
        "intent_candidates": [],
        "threshold": None,
        "ood_threshold": None,
        "router_revision": ROUTER_REVISION,
        "reason": reason,
    }


def _render_diagnostics(
    route: dict[str, Any],
    calls: tuple[ToolCall, ...],
    results: tuple[dict[str, Any], ...],
    response_path: str,
) -> str:
    candidates = route.get("intent_candidates")
    candidate_text = (
        "\n".join(
            f"- `{item.get('intent')}`: {float(item.get('probability', 0)):.3f}"
            for item in candidates
            if isinstance(item, dict)
        )
        if isinstance(candidates, list)
        else ""
    )
    call_text = (
        "\n".join(
            f"- `{call.name}` `{json.dumps(call.arguments, sort_keys=True)}`"
            for call in calls
        )
        or "- None"
    )
    result_text = (
        "\n".join(
            f"- `{item.get('name')}`: {'success' if item.get('ok') else 'error'}"
            for item in results
        )
        or "- None"
    )
    return (
        "### Experiment diagnostics\n\n"
        f"- Route: `{route.get('route')}`\n"
        f"- In-domain probability: `{route.get('banking_probability')}`\n"
        f"- OOD probability: `{route.get('ood_probability')}`\n"
        f"- Response path: `{response_path}`\n\n"
        f"**Top intents**\n{candidate_text or '- None'}\n\n"
        f"**9B tool calls**\n{call_text}\n\n"
        f"**Tool results**\n{result_text}\n\n"
        f"Model revision: `{MODEL_REVISION[:12]}…`"
    )


with gr.Blocks(
    title="Retail Bank Customer Service POC",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber"),
) as demo:
    gr.Markdown(
        """
        # Retail Bank Customer Service POC

        <div class="synthetic-banner">
        <strong>Fictional data only.</strong> The dual-head classifier gates
        high-confidence OOD requests and supplies intent guidance. The 9B model owns
        allowed conversation, tool selection, tool arguments, and final responses.
        No real banking system is connected.
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            profile_panel = gr.HTML()
            snapshot_panel = gr.Markdown()
            activity_panel = gr.Markdown()
            diagnostics_panel = gr.Markdown()
            with gr.Row():
                refresh_button = gr.Button("Refresh state", size="sm")
                reset_button = gr.Button("Reset demo", size="sm")
            gr.Button("Log out", link="/logout?all_session=false", size="sm")
        with gr.Column(scale=2, min_width=580):
            chatbot = gr.Chatbot(
                height=590,
                layout="bubble",
                type="messages",
                placeholder="Talk naturally to the 9B synthetic bank agent.",
            )
            with gr.Row():
                message_box = gr.Textbox(
                    placeholder="Ask the signed-in synthetic bank agent.",
                    show_label=False,
                    scale=8,
                    submit_btn=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(
                examples=[[prompt] for prompt in PRESETS],
                inputs=message_box,
                label="Preset evaluation prompts",
            )

    conversation_history = gr.State([])
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
        inputs=[message_box, chatbot, conversation_history, session_epoch],
        outputs=[
            message_box,
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            diagnostics_panel,
            pending_turn,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name="chat",
        api_description="CPU OOD dispatch followed by a registered 9B ZeroGPU event.",
        queue=True,
        trigger_mode="once",
    )
    model_event = pending_turn.change(
        finalize_turn,
        inputs=[
            pending_turn,
            session_epoch,
            chatbot,
            conversation_history,
        ],
        outputs=[
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            diagnostics_panel,
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
        inputs=[
            pending_turn,
            session_epoch,
            chatbot,
            conversation_history,
        ],
        outputs=[
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            diagnostics_panel,
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
        outputs=[profile_panel, snapshot_panel, activity_panel, diagnostics_panel],
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
            conversation_history,
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
