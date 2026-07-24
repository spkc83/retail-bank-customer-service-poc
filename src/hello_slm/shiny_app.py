from __future__ import annotations

import asyncio
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from hello_slm.chat_presets import ARITHMETIC_CHAT_PRESETS, ChatPreset
from hello_slm.chat_runtime import ArithmeticChatRuntime, ChatReply
from hello_slm.training import PipelineError

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "arithmetic-curriculum-30m.toml"
CHECKPOINT_PATH = (
    REPO_ROOT / "artifacts" / "arithmetic-curriculum-30m" / "checkpoints" / "latest.pt"
)
MODEL_RUNTIME = ArithmeticChatRuntime(CONFIG_PATH, CHECKPOINT_PATH)


def _preset_button(preset: ChatPreset) -> ui.Tag:
    support_class = "preset-supported" if preset.supported else "preset-exploratory"
    badge = "SUPPORTED" if preset.supported else "LIMIT CHECK"
    return ui.input_action_button(
        f"preset_{preset.id}",
        ui.div(
            ui.span(preset.label, class_="preset-label"),
            ui.span(preset.prompt, class_="preset-expression"),
            ui.span(f"Expected: {preset.expected_answer}", class_="preset-expected"),
            ui.span(badge, class_=f"preset-badge {support_class}"),
        ),
        class_="preset-button",
        width="100%",
        title=f"Run {preset.operation} preset",
    )


WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "**Arithmetic Lab is ready.** Type a bounded arithmetic question or choose a "
        "preset. Each turn is evaluated independently with deterministic decoding."
    ),
}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.div("H", class_="brand-mark"),
            ui.div(
                ui.h2("Hello SLM", class_="brand-title"),
                ui.p("Arithmetic checkpoint lab", class_="brand-subtitle"),
            ),
            class_="brand-lockup",
        ),
        ui.p(
            "Run known cases against the locally trained 27.8M-parameter model.",
            class_="sidebar-intro",
        ),
        ui.h3("Preset test cases", class_="section-label"),
        ui.div(*(_preset_button(preset) for preset in ARITHMETIC_CHAT_PRESETS), class_="presets"),
        ui.input_action_button(
            "clear_chat",
            "Clear conversation",
            class_="clear-button",
            width="100%",
        ),
        ui.tags.details(
            ui.tags.summary("Model limitations"),
            ui.tags.ul(
                ui.tags.li(
                    "Closed-corpus paraphrase and fact recall—not unseen arithmetic reasoning."
                ),
                ui.tags.li("Addition, subtraction, and exact division are the supported claims."),
                ui.tags.li("Multiplication is exploratory and scored 30% in evaluation."),
                ui.tags.li("Responses may be wrong, repetitive, or reproduce training text."),
                ui.tags.li("No moderation, human review, abuse monitoring, or external tools."),
                ui.tags.li("Not for medical, legal, financial, emergency, or security decisions."),
            ),
            class_="limitations",
        ),
        width=330,
        open="desktop",
        class_="lab-sidebar",
    ),
    ui.include_css(PACKAGE_DIR / "arithmetic_chat.css", method="inline"),
    ui.div(
        ui.div(
            ui.div(
                ui.span("LOCAL CHECKPOINT", class_="eyebrow"),
                ui.h1("Arithmetic Tutor Lab"),
                ui.p(
                    "A focused interface for probing the model exactly as trained—one "
                    "question at a time."
                ),
            ),
            ui.div(
                ui.span(class_="status-dot"),
                ui.output_text("runtime_status", inline=True),
                class_="runtime-pill",
            ),
            class_="hero",
        ),
        ui.layout_columns(
            ui.value_box(
                "Model",
                "27.8M",
                ui.span("parameters", class_="metric-note"),
                showcase="◫",
                theme="primary",
            ),
            ui.value_box(
                "Vocabulary",
                "156",
                ui.span("realized tokens", class_="metric-note"),
                showcase="Aa",
                theme="teal",
            ),
            ui.value_box(
                "Supported exact match",
                "95.10%",
                ui.span("closed-corpus test", class_="metric-note"),
                showcase="✓",
                theme="success",
            ),
            col_widths=(4, 4, 4),
            fill=False,
            class_="metric-grid",
        ),
        ui.card(
            ui.card_header(
                ui.div(
                    ui.div(
                        ui.h2("Test conversation"),
                        ui.p("Greedy decoding · 64-token response budget · single-turn context"),
                    ),
                    ui.output_text("last_run", inline=True),
                    class_="chat-heading",
                )
            ),
            ui.chat_ui(
                "model_chat",
                messages=[WELCOME_MESSAGE],
                placeholder="Ask: What is 42 + 19?",
                width="100%",
                height="100%",
                fill=True,
            ),
            class_="chat-card",
            min_height="540px",
        ),
        class_="app-content",
    ),
    title="Hello SLM · Arithmetic Tutor Lab",
    fillable=True,
    class_="lab-page",
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    chat = ui.Chat("model_chat", on_error="sanitize")
    runtime_status_value = reactive.Value("Checkpoint idle")
    last_run_value = reactive.Value("No prompt run yet")

    @output
    @render.text
    def runtime_status() -> str:
        return runtime_status_value()

    @output
    @render.text
    def last_run() -> str:
        return last_run_value()

    @chat.on_user_submit
    async def handle_user_input(user_input: str) -> None:
        runtime_status_value.set("Running inference…")
        last_run_value.set("Working…")
        try:
            reply = await asyncio.to_thread(MODEL_RUNTIME.reply, user_input)
        except (PipelineError, OSError, ValueError) as exc:
            runtime_status_value.set("Inference unavailable")
            last_run_value.set("Prompt failed")
            await chat.append_message(f"**Could not run the model:** {exc}")
            return
        except Exception:
            runtime_status_value.set("Runtime error")
            last_run_value.set("Prompt failed")
            await chat.append_message(
                "**The local inference runtime failed unexpectedly.** "
                "Check the server log for details."
            )
            raise
        await chat.append_message(reply.response)
        runtime_status_value.set(f"Checkpoint ready · {reply.device}")
        last_run_value.set(_reply_summary(reply))

    def register_preset(preset: ChatPreset) -> None:
        button = getattr(input, f"preset_{preset.id}")

        @reactive.effect
        @reactive.event(button)
        def submit_preset() -> None:
            chat.update_user_input(value=preset.prompt, submit=True, focus=True)

    for preset in ARITHMETIC_CHAT_PRESETS:
        register_preset(preset)

    @reactive.effect
    @reactive.event(input.clear_chat)
    async def clear_chat() -> None:
        await chat.clear_messages()
        await chat.append_message(WELCOME_MESSAGE)
        last_run_value.set("No prompt run yet")


def _reply_summary(reply: ChatReply) -> str:
    return f"Step {reply.global_step:,} · {reply.latency_seconds:.2f}s · {reply.device}"


app = App(app_ui, server)
