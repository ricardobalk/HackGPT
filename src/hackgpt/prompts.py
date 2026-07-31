"""Load and compose setup prompts."""

import json
from datetime import date
from importlib.resources import files
from typing import cast

from hackgpt.models import PromptCategory

CUSTOM_PROMPT = (
    "You are an AI assistant helping a user with a task. "
    "Your tone of voice is friendly and helpful."
)


def load_setup_prompts() -> dict[str, PromptCategory]:
    """Load bundled prompt categories and append the custom prompt option."""
    prompt_file = files("hackgpt.data").joinpath("setup_prompts.json")
    prompts = cast(
        dict[str, PromptCategory],
        json.loads(prompt_file.read_text(encoding="utf-8")),
    )
    prompts["custom"] = {
        "label": "Custom",
        "options": {
            "custom": {
                "label": "Custom",
                "value": CUSTOM_PROMPT,
            }
        },
    }
    return prompts


def add_historical_context(prompt: str, cutoff: date, today: date) -> str:
    """Append a requested historical knowledge perspective to a prompt."""
    context = (
        "Answer as though your knowledge were limited to information available on "
        f"{cutoff.strftime('%B %d, %Y')}. "
        f"The user's stated current date is {today.strftime('%B %d, %Y')}."
    )
    return f"{prompt} {context}"
