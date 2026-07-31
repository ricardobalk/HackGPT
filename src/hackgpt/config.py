"""Static application configuration and model metadata."""

from datetime import date
from typing import Final, Literal, TypedDict


class PageConfig(TypedDict):
    """Typed subset of Streamlit page configuration used by HackGPT."""

    page_title: str
    page_icon: str
    layout: Literal["wide"]


APP_NAME: Final = "HackGPT"
APP_VERSION: Final = "1.0.0-alpha.2"
RELEASE_DATE: Final = date(2026, 7, 31)

PAGE_CONFIG: Final[PageConfig] = {
    "page_title": APP_NAME,
    "page_icon": "🤖",
    "layout": "wide",
}

MIN_CUTOFF_DATE: Final = date(1950, 1, 1)
DEFAULT_CUTOFF_DATE: Final = date(date.today().year, 1, 1)
MAX_CUTOFF_DATE: Final = date.today()

APP_FOOTER: Final = (
    f"{APP_NAME} {APP_VERSION}, released on {RELEASE_DATE.strftime('%B %d, %Y')}. "
    f"Keep in mind that {APP_NAME} may produce inaccurate information about people, "
    "places, or facts."
)
