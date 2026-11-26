"""Reasoning effort a request asks for, in OpenAI's vocabulary.

Every dialect is reduced to one of LEVELS (or None: the server's setting).
Pure functions — no FastAPI, no model code.
"""

LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Anthropic budget_tokens -> level: under 4k is a short think, 16k and up a
# long one. Anthropic's own minimum budget is 1024.
BUDGET_LEVELS = ((4096, "low"), (16384, "medium"))


class ReasoningError(ValueError):
    """The request's reasoning setting is not one we understand (a 400)."""


def level(value, field="reasoning_effort"):
    """Validate an OpenAI-style effort; None means not requested."""
    if value is None:
        return None
    if value not in LEVELS:
        raise ReasoningError(f"{field} must be one of {', '.join(LEVELS)}")
    return value


def budget_level(budget):
    for below, name in BUDGET_LEVELS:
        if budget < below:
            return name
    return "high"


def from_anthropic(thinking):
    """Anthropic `thinking` -> effort level, or None for the model default."""
    if thinking is None:
        return None
    if not isinstance(thinking, dict):
        raise ReasoningError("thinking must be an object")
    kind = thinking.get("type")
    if kind == "disabled":
        return "none"
    if kind == "adaptive":
        return None
    if kind != "enabled":
        raise ReasoningError("thinking.type must be enabled, disabled or adaptive")
    budget = thinking.get("budget_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ReasoningError("thinking.budget_tokens must be a positive integer")
    return budget_level(budget)
