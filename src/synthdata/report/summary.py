"""One wording for the summary claims, shared by the HTML page, the app and the CLI."""

from __future__ import annotations


def twin_status(metrics: dict) -> tuple[str, str]:
    """(kind, text); kind is one of none, untestable, ok, flagged. A report without `twin_testable` is legacy: testable."""
    flagged = metrics.get("twin_flagged")
    if flagged is None:
        return "none", "twin: not applicable"
    if metrics.get("twin_testable", True) is False:
        return "untestable", "twin: not testable (no condition reaches significance)"
    if flagged:
        return "flagged", "twin flagged \u2717"
    return "ok", "twin indistinguishable \u2713"
