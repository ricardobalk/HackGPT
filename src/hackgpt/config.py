"""Static application configuration and model metadata."""

from datetime import date
from typing import Final

APP_NAME: Final = "HackGPT"
APP_VERSION: Final = "1.0.0-alpha.2"
RELEASE_DATE: Final = date(2026, 7, 31)

PAGE_CONFIG: Final = {
    "page_title": APP_NAME,
    "page_icon": "🤖",
    "layout": "wide",
}

MODEL_DESCRIPTIONS: Final = {
    "gpt-5.6-terra": (
        "Balanced intelligence, latency, and cost. Recommended for general use."
    ),
    "gpt-5.6-sol": (
        "Flagship model for the most complex reasoning and professional work."
    ),
    "gpt-5.6-luna": (
        "Cost-efficient model for simpler and high-volume workloads."
    ),
    "gpt-4.1": (
        "High-intelligence, low-latency model with strong instruction following."
    ),
    "gpt-4.1-mini": (
        "Smaller, faster GPT-4.1 model for cost-sensitive workloads."
    ),
    "gpt-4o": (
        "Legacy multimodal GPT-4o model. Deprecated; prefer a newer model."
    ),
    "gpt-4o-mini": (
        "Fast, affordable multimodal model for focused tasks."
    ),
}

MODELS_WITH_ADVANCED_REASONING: Final = frozenset(
    {"gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"}
)
MODELS_WITH_VERBOSITY: Final = MODELS_WITH_ADVANCED_REASONING
REASONING_EFFORTS: Final = ("none", "low", "medium", "high", "xhigh", "max")
REASONING_MODES: Final = ("standard", "pro")
VERBOSITY_LEVELS: Final = ("low", "medium", "high")
MIN_CUTOFF_DATE: Final = date(1950, 1, 1)
DEFAULT_CUTOFF_DATE: Final = date(date.today().year, 1, 1)
MAX_CUTOFF_DATE: Final = date.today()

APP_FOOTER: Final = (
    f"{APP_NAME} {APP_VERSION}, released on {RELEASE_DATE.strftime('%B %d, %Y')}. "
    f"Keep in mind that {APP_NAME} may produce inaccurate information about people, "
    "places, or facts."
)
