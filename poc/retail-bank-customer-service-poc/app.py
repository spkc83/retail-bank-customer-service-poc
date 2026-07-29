"""Authenticated model-driven synthetic retail-bank conversational POC."""

# ZeroGPU must patch PyTorch before the CPU router imports it.
# ruff: noqa: I001

from __future__ import annotations

import html
import hashlib
import json
import os
from typing import Any

if os.environ.get("POC_SKIP_MODEL_LOAD") == "1":
    from zero_gpu_runtime import (
        MODEL_REVISION,
        MODEL_ID,
        count_tokens,
        generate_text,
        runtime_metadata,
        spaces_runtime as spaces,
    )
else:
    import spaces

    from zero_gpu_runtime import (
        MODEL_ID,
        MODEL_REVISION,
        count_tokens,
        generate_text,
        runtime_metadata,
    )

import gradio as gr

from auth import load_demo_auth
from model_service import (
    AgentExecutionError,
    AgentProtocolError,
    ConversationalBankingAgent,
    ModelPassTrace,
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
AUTH_MESSAGE = (
    "<strong>Synthetic demo only.</strong><br>"
    "Use either public test account:<br>"
    + "<br>".join(
        f"<code>{html.escape(username)}</code> / <code>{html.escape(password)}</code>"
        for username, password in AUTH_CREDENTIALS
    )
)
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

    def runtime_metadata(self) -> dict[str, str]:
        return runtime_metadata()


@spaces.GPU(size="large", duration=90)
def run_model_turn(
    message: str,
    visible_history: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[
    Any,
    list[dict[str, str]],
    list[dict[str, Any]],
    str,
    str,
    str,
    Any,
    Any,
    Any,
]:
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

    agent = ConversationalBankingAgent(bank=BANK, model=_RuntimeModel())
    try:
        result = agent.run_turn(
            username=username,
            session_hash=session_hash,
            message=message,
            conversation=conversation,
            router_result=route,
        )
    except AgentExecutionError as error:
        failed_conversation = [
            *error.conversation,
            {"role": "assistant", "content": MODEL_FAILURE_RESPONSE},
        ]
        enabled = gr.update(interactive=True)
        return (
            gr.update(value="", interactive=True),
            _visible_from_conversation(failed_conversation),
            failed_conversation,
            render_snapshot(error.snapshot),
            (
                "The 9B model failed after executing the tool calls shown in "
                "diagnostics. No CPU-authored servicing answer was substituted."
            ),
            (
                f"{_render_diagnostics(
                    route,
                    error.tool_calls,
                    error.tool_results,
                    '9B second-pass failure',
                    error.model_passes,
                )}\n\n"
                f"Failure type: `{type(error.__cause__).__name__}`"
            ),
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
            gr.update(value="", interactive=True),
            _visible_from_conversation(failed_conversation),
            failed_conversation,
            render_snapshot(BANK.snapshot(username, session_hash)),
            (
                "The 9B model event failed. No CPU-authored servicing answer was "
                "substituted; the synthetic dashboard shows the current backend state."
            ),
            (
                f"{_render_diagnostics(route, (), (), '9B model failure')}\n\n"
                f"Failure type: `{type(error).__name__}`"
            ),
            enabled,
            enabled,
            enabled,
        )

    enabled = gr.update(interactive=True)
    return (
        gr.update(value="", interactive=True),
        _visible_from_conversation(result.conversation),
        result.conversation,
        render_snapshot(result.snapshot),
        (
            (
                "The 9B model selected and called the synthetic tools, then authored "
                "the final response from their results."
            )
            if result.tool_calls
            else (
                "The 9B model authored the base answer, which was retained after "
                "the labeled reflection pass."
            )
        ),
        _render_diagnostics(
            route,
            result.tool_calls,
            result.tool_results,
            f"9B {result.response_path}",
            result.model_passes,
            result.response,
        ),
        enabled,
        enabled,
        enabled,
    )


def fail_model_turn(
    message: str,
    current_conversation: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[
    Any,
    list[dict[str, str]],
    list[dict[str, Any]],
    str,
    str,
    str,
    Any,
    Any,
    Any,
]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    username, session_hash = _identity(request)
    conversation = canonical_conversation(current_conversation)
    enabled = gr.update(interactive=True)
    failed_conversation = _with_text_turn(
        conversation,
        message,
        MODEL_FAILURE_RESPONSE,
    )
    return (
        gr.update(value="", interactive=True),
        _visible_from_conversation(failed_conversation),
        failed_conversation,
        render_snapshot(BANK.snapshot(username, session_hash)),
        (
            "ZeroGPU could not allocate or complete the 9B model turn. "
            "No CPU-authored servicing answer was substituted."
        ),
        _render_diagnostics(
            _uncertain_route("GPU event failed before routing completed"),
            (),
            (),
            "ZeroGPU/model unavailable",
        ),
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
        "Ready. Allowed turns are handled by the 9B model; OOD is gated.",
        "### Experiment diagnostics\n\nNo turn has run yet.",
    )


def refresh_snapshot(request: gr.Request) -> str:
    username, session_hash = _identity(request)
    return render_snapshot(BANK.snapshot(username, session_hash))


def reset_session(
    request: gr.Request,
) -> tuple[list[Any], list[Any], str, str, Any, Any, Any, Any]:
    username, session_hash = _identity(request)
    snapshot = BANK.reset(username, session_hash)
    enabled = gr.update(interactive=True)
    return (
        [],
        [],
        render_snapshot(snapshot),
        "This browser session and complete model conversation were reset.",
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
) -> tuple[
    Any,
    list[dict[str, str]],
    list[dict[str, Any]],
    str,
    str,
    str,
    Any,
    Any,
    Any,
]:
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
    model_passes: tuple[ModelPassTrace, ...] = (),
    visible_response: str | None = None,
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
            (
                f"- `{call.id}` `{call.name}` "
                f"`{json.dumps(call.arguments, sort_keys=True)}`"
            )
            for call in calls
        )
        or "- None"
    )
    result_text = (
        "\n".join(
            (
                f"- `{call.id}` `{call.name}`: "
                f"{'success' if result.get('ok') else 'error'}"
                f"{_diagnostic_error_suffix(result)}"
            )
            for call, result in zip(calls, results, strict=False)
        )
        or "- None"
    )
    pass_text = (
        "\n".join(
            (
                f"- `{item.label}` — input tokens `{item.input_tokens}`, "
                f"prompt SHA-256 `{item.prompt_sha256}`, output characters "
                f"`{item.output_chars}`, raw output SHA-256 `{item.output_sha256}`; "
                f"Runtime device: `{item.runtime_device}`; CUDA device name: "
                f"`{item.cuda_device_name}`\n"
                f"<details><summary>Show {item.label} raw output</summary>"
                f"<pre><code>{html.escape(item.raw_output)}</code></pre></details>"
            )
            for item in model_passes
        )
        or "- None; the 9B generator was not invoked."
    )
    visible_hash = (
        hashlib.sha256(visible_response.encode("utf-8")).hexdigest()
        if isinstance(visible_response, str)
        else "not recorded"
    )
    space_commit = os.environ.get("SPACE_COMMIT_SHA", "unavailable")
    return (
        "### Experiment diagnostics\n\n"
        f"- Route: `{route.get('route')}`\n"
        f"- In-domain probability: `{route.get('banking_probability')}`\n"
        f"- OOD probability: `{route.get('ood_probability')}`\n"
        f"- Conversation context applied: `{route.get('context_applied', False)}`\n"
        f"- Context reason: `{route.get('context_reason')}`\n"
        f"- Response path: `{response_path}`\n\n"
        f"**Top intents**\n{candidate_text or '- None'}\n\n"
        f"**9B tool calls**\n{call_text}\n\n"
        f"**Tool results**\n{result_text}\n\n"
        f"**9B generation provenance**\n{pass_text}\n\n"
        f"- Generation calls: `{len(model_passes)}`\n"
        f"- Model: `{MODEL_ID}`\n"
        f"- Exact model revision: `{MODEL_REVISION}`\n"
        f"- Space commit: `{space_commit}`\n"
        f"- Registered execution boundary: `ZeroGPU large`\n"
        f"- Visible response SHA-256: `{visible_hash}`"
    )


def _diagnostic_error_suffix(result: dict[str, Any]) -> str:
    if result.get("ok"):
        return ""
    error = result.get("error")
    if not isinstance(error, dict):
        return ""
    code = error.get("code")
    return f" (`{code}`)" if isinstance(code, str) and code else ""


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
        OOD requests and supplies intent guidance. The 9B model owns
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
    model_event = gr.on(
        triggers=[message_box.submit, send_button.click],
        fn=run_model_turn,
        inputs=[message_box, chatbot, conversation_history],
        outputs=[
            message_box,
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            diagnostics_panel,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name="chat",
        api_description="Direct dual-head routing and 9B ZeroGPU conversational turn.",
        queue=True,
        trigger_mode="once",
    )
    model_event.failure(
        fail_model_turn,
        inputs=[
            message_box,
            conversation_history,
        ],
        outputs=[
            message_box,
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            diagnostics_panel,
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
        outputs=[
            chatbot,
            conversation_history,
            snapshot_panel,
            activity_panel,
            message_box,
            send_button,
            reset_button,
            refresh_button,
        ],
        api_name="reset_demo",
        queue=False,
        cancels=[model_event],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        auth=AUTH_CREDENTIALS,
        auth_message=AUTH_MESSAGE,
        ssr_mode=False,
        show_error=False,
    )
