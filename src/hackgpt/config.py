"""Static application configuration and model metadata."""

from datetime import date
from typing import Final

APP_NAME: Final = "HackGPT"
APP_VERSION: Final = "1.0.0-alpha.1"
RELEASE_DATE: Final = date(2023, 7, 8)

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
}

REASONING_EFFORTS: Final = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_CUTOFF_DATE: Final = date(2021, 9, 1)

APP_FOOTER: Final = (
    f"{APP_NAME} {APP_VERSION}, released on {RELEASE_DATE.strftime('%B %d, %Y')}. "
    f"Keep in mind that {APP_NAME} may produce inaccurate information about people, "
    "places, or facts."
)
