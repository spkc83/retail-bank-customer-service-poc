"""Authenticated, model-driven synthetic retail-bank customer-service POC."""

from __future__ import annotations

import html
import os
from typing import Any

import gradio as gr

from auth import load_demo_auth
from model_service import ToolCallError
from policy import (
    MODEL_FAILURE_RESPONSE,
    OOD_RESPONSE,
    SENSITIVE_RESPONSE,
    contains_sensitive_value,
)
from router import ROUTER_REVISION, LearnedBankingRouter
from state import BANK
from zero_gpu_runtime import run_model_service

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
    "Show my account balances.",
    "Show my five most recent transactions.",
    "What is the status of my debit card?",
    "My card was stolen. Freeze it.",
    "Please replace my debit card.",
    "I did not make the latest card purchase. Dispute it.",
    "Show my recent transfers.",
    "Cancel my pending transfer.",
    "Show my open service cases.",
    "What is the weather tomorrow?",
]


def respond(
    message: str,
    history: list[dict[str, Any]],
    request: gr.Request,
) -> tuple[str, str, str]:
    username, session_hash = _identity(request)
    if contains_sensitive_value(message):
        return (
            SENSITIVE_RESPONSE,
            render_snapshot(BANK.snapshot(username, session_hash)),
            "🛡️ Credential guard blocked the request before routing or GPU allocation.",
        )
    route = route_query(message, history)
    if route["route"] != "in_domain":
        return (
            OOD_RESPONSE,
            render_snapshot(BANK.snapshot(username, session_hash)),
            (
                "🧭 CPU domain router refused the request before ZeroGPU allocation "
                f"(banking probability {route.get('banking_probability')!s})."
            ),
        )
    try:
        result = run_model_service(username, session_hash, message, history)
    except (ToolCallError, RuntimeError, ValueError):
        return (
            MODEL_FAILURE_RESPONSE,
            render_snapshot(BANK.snapshot(username, session_hash)),
            "⚠️ Model output failed tool validation; no unvalidated action was executed.",
        )
    response = (
        f"{result['response']}\n\n"
        f"---\n_Model tool: `{result['tool_name']}` · "
        f"model revision `{str(result['model_revision'])[:8]}…`_"
    )
    return (
        response,
        render_snapshot(result["snapshot"]),
        (
            f"✅ 9B model proposed `{result['tool_name']}`; the policy validated it, "
            "the session database executed it, and the model wrote the final answer."
        ),
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


def render_snapshot(snapshot: dict[str, Any]) -> str:
    accounts = "\n".join(
        (
            f"- **{account['name']} ····{account['last4']}** — "
            f"{_money(account['available_balance_cents'], account['currency'])} available"
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
        proposes constrained backend tools, and writes the final response. No real banking
        system is connected.
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
            chat_interface = gr.ChatInterface(
                fn=respond,
                type="messages",
                chatbot=chatbot,
                additional_outputs=[snapshot_panel, activity_panel],
                examples=PRESETS,
                cache_examples=False,
                example_labels=[
                    "Balances",
                    "Transactions",
                    "Card status",
                    "Freeze stolen card",
                    "Replace card",
                    "Dispute purchase",
                    "Transfers",
                    "Cancel transfer",
                    "Service cases",
                    "OOD check",
                ],
                title=None,
                description=None,
                api_name="chat",
                api_description=(
                    "Authenticated model-driven synthetic retail-bank servicing chat."
                ),
            )

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
        reset_demo,
        outputs=[chatbot, snapshot_panel, activity_panel],
        api_name="reset_demo",
        queue=False,
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
