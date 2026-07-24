from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from hello_slm.banking_chat_runtime import BankingChatReply, BankingChatRuntime
from hello_slm.banking_hf_generator import (
    BANKING_MODEL_ENV,
    DEFAULT_BANKING_MODEL_PATH,
    HuggingFaceBankingGenerator,
    MissingBankingCheckpointError,
)
from hello_slm.banking_policy import OOD_STOCK_RESPONSE

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
MODEL_PATH = Path(
    os.environ.get(BANKING_MODEL_ENV, str(REPO_ROOT / DEFAULT_BANKING_MODEL_PATH))
).expanduser()
MODEL_RUNTIME = BankingChatRuntime(generator=HuggingFaceBankingGenerator(MODEL_PATH))


@dataclass(frozen=True)
class BankingPreset:
    id: str
    label: str
    prompt: str
    expected_route: str


BANKING_CHAT_PRESETS = (
    BankingPreset(
        id="card_replacement",
        label="Card support",
        prompt="I lost my debit card. What should I do?",
        expected_route="in-domain",
    ),
    BankingPreset(
        id="follow_up_fee",
        label="Follow-up",
        prompt="What about the fee?",
        expected_route="in-domain after card context",
    ),
    BankingPreset(
        id="transfer_limit",
        label="Transfers",
        prompt="How do I raise my daily transfer limit?",
        expected_route="in-domain",
    ),
    BankingPreset(
        id="weather_ood",
        label="OOD refusal",
        prompt="What is the weather tomorrow?",
        expected_route="OOD stock response",
    ),
)

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "**Banking v2 lab is ready.** Use presets or ask a retail-banking support "
        "question. OOD queries are rejected before model generation."
    ),
}


def _preset_button(preset: BankingPreset) -> ui.Tag:
    return ui.input_action_button(
        f"banking_preset_{preset.id}",
        ui.div(
            ui.strong(preset.label),
            ui.div(preset.prompt),
            ui.tags.small(f"Expected: {preset.expected_route}"),
        ),
        width="100%",
        class_="btn-outline-primary",
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("Hello SLM Banking"),
        ui.p("Retail banking multi-turn checkpoint lab."),
        ui.input_action_button("clear_chat", "Clear conversation", width="100%"),
        ui.hr(),
        ui.h4("Preset test cases"),
        *(_preset_button(preset) for preset in BANKING_CHAT_PRESETS),
        ui.hr(),
        ui.tags.details(
            ui.tags.summary("Runtime contract"),
            ui.tags.ul(
                ui.tags.li("OOD queries return the exact stock response before generation."),
                ui.tags.li("In-domain queries require a local trained Transformers checkpoint."),
                ui.tags.li(f"Default checkpoint: {MODEL_PATH}"),
                ui.tags.li(f"Override with {BANKING_MODEL_ENV}."),
            ),
        ),
        width=340,
        open="desktop",
    ),
    ui.h1("Banking Services Chat Lab"),
    ui.layout_columns(
        ui.value_box("Route", ui.output_text("route_status", inline=True)),
        ui.value_box("Confidence", ui.output_text("confidence_status", inline=True)),
        ui.value_box("Candidates", ui.output_text("candidate_status", inline=True)),
        col_widths=(4, 4, 4),
    ),
    ui.card(
        ui.card_header("Multi-turn chat"),
        ui.chat_ui(
            "banking_chat",
            messages=[WELCOME_MESSAGE],
            placeholder="Ask about cards, accounts, transfers, payments, or loans.",
            width="100%",
            height="560px",
        ),
    ),
    ui.tags.p(ui.tags.strong("OOD stock response: "), OOD_STOCK_RESPONSE),
    title="Hello SLM · Banking Services Lab",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    chat = ui.Chat("banking_chat", on_error="sanitize")
    route_value = reactive.Value("idle")
    confidence_value = reactive.Value("n/a")
    candidate_value = reactive.Value("0")

    @output
    @render.text
    def route_status() -> str:
        return route_value()

    @output
    @render.text
    def confidence_status() -> str:
        return confidence_value()

    @output
    @render.text
    def candidate_status() -> str:
        return candidate_value()

    @chat.on_user_submit
    async def handle_user_input(user_input: str) -> None:
        route_value.set("routing")
        confidence_value.set("...")
        candidate_value.set("0")
        try:
            reply = await asyncio.to_thread(
                MODEL_RUNTIME.reply,
                _session_id(session),
                user_input,
            )
        except MissingBankingCheckpointError as exc:
            route_value.set("in_domain")
            confidence_value.set("checkpoint missing")
            candidate_value.set("0")
            await chat.append_message(f"**Banking checkpoint unavailable:** {exc}")
            return
        except Exception:
            route_value.set("runtime_error")
            confidence_value.set("failed")
            candidate_value.set("0")
            await chat.append_message("**Banking inference failed.** Check the server log.")
            raise

        await chat.append_message(reply.response)
        _set_status(reply, route_value, confidence_value, candidate_value)

    def register_preset(preset: BankingPreset) -> None:
        button = getattr(input, f"banking_preset_{preset.id}")

        @reactive.effect
        @reactive.event(button)
        def submit_preset() -> None:
            chat.update_user_input(value=preset.prompt, submit=True, focus=True)

    for preset in BANKING_CHAT_PRESETS:
        register_preset(preset)

    @reactive.effect
    @reactive.event(input.clear_chat)
    async def clear_chat() -> None:
        await chat.clear_messages()
        await chat.append_message(WELCOME_MESSAGE)
        route_value.set("idle")
        confidence_value.set("n/a")
        candidate_value.set("0")


def _session_id(session: Session) -> str:
    return str(getattr(session, "id", id(session)))


def _set_status(
    reply: BankingChatReply,
    route_value: reactive.Value[str],
    confidence_value: reactive.Value[str],
    candidate_value: reactive.Value[str],
) -> None:
    route_value.set(reply.route)
    confidence_value.set(f"{reply.domain_confidence:.2f}")
    candidate_value.set(str(len(reply.candidates)))


app = App(app_ui, server)
