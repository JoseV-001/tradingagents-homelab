"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader."""

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager."""

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call (such as "
            "when key analysts report Insufficient Data and remaining verified "
            "signals are mixed or non-directional)."
        ),
    )
    rationale: str = Field(
        description=(
            "Summary of the verified evidence from both sides of the debate, "
            "ending with which verified facts led to the recommendation. "
            "Explicitly state any analyst dimensions that reported Insufficient "
            "Data (DATA_UNAVAILABLE), discard any debater claim not backed by "
            "verified reports, and NEVER invent price targets, supports, "
            "resistances, or stop-losses."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including sizing guidance relative to a standard allocation. The "
            "research team does not see the caller's holdings; the trader and "
            "portfolio manager apply the actual position. Never invent arbitrary "
            "price targets, supports, resistances, or stop-losses; cite price "
            "levels only if derived via an explicit arithmetic formula from "
            "verified market-snapshot figures, otherwise state explicitly that "
            "there is no validated technical price level."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader."""

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored strictly in verified analyst "
            "reports and the research plan (two to five sentences). Preserve "
            "any Insufficient Data (DATA_UNAVAILABLE) uncertainties explicitly. "
            "If entry_price or stop_loss is provided, show the exact arithmetic "
            "formula from verified snapshot figures (e.g. Close and ATR); "
            "otherwise state explicitly that there is no validated technical "
            "entry, support, resistance, or stop-loss level."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. "
            "MUST be null (omitted) when action is Hold or when no explicit, "
            "verifiable level from the verified market snapshot applies. "
            "NEVER invent or arbitrate unverified support/resistance entry prices."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. MUST be null (omitted) "
            "when action is Hold or unless derived from an explicit arithmetic "
            "calculation on verified snapshot figures (such as Close +/- k*ATR, "
            "shown step-by-step in reasoning). NEVER invent arbitrary stop-losses."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Optional sizing guidance, e.g. '0% new exposure (maintain current)' "
            "or '2-3% of portfolio (reduced due to Insufficient Data)'."
        ),
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown."""
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager."""

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise, 100% evidence-traceable action plan covering direction, "
            "position sizing, verified risk parameters, and data limitations "
            "(two to four sentences). Explicitly preserve uncertainties where "
            "analysts reported Insufficient Data (DATA_UNAVAILABLE), and state "
            "explicitly when no validated technical support, resistance, stop, "
            "or price target exists."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning 100% traceable to specific verified evidence "
            "from each analyst report (Market, Sentiment, News, Fundamentals). "
            "Explicitly list every analyst report that returned Insufficient "
            "Data (DATA_UNAVAILABLE) and explain how that uncertainty constrains "
            "conviction and sizing. Never invent company facts, macro data, "
            "supports, resistances, stop-losses, or price targets. If prior "
            "lessons are referenced in the prompt context, incorporate them; "
            "otherwise rely solely on the current verified analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Optional target price in the instrument's quote currency. MUST be "
            "null (omitted) unless derived via an explicit, reproducible "
            "mathematical formula from verified report data shown step-by-step "
            "in investment_thesis. NEVER invent, guess, or arbitrate a price "
            "target from unverified multiples, moving averages, or Bollinger bands."
        ),
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding or review period, e.g. 'Reavaliar apÃ³s divulgaÃ§Ã£o de dados verificÃ¡veis'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects."""
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst."""

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"
    INSUFFICIENT_DATA = "Insufficient Data"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst."""

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish / Insufficient Data. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity from 0 to 10. "
            "Use null when there is insufficient source evidence to calculate "
            "a meaningful sentiment score. Never use 5 merely because data "
            "is unavailable."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a structured sentiment report with explicit missing-score support."""
    score = (
        f"{report.overall_score:.1f}/10"
        if report.overall_score is not None
        else "N/A"
    )
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {score})",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# News Analyst
# ---------------------------------------------------------------------------


class NewsCoverageBand(StrEnum):
    """Deterministic evidence-availability band for the News Analyst."""

    SUFFICIENT_DATA = "Sufficient Data"
    PARTIAL_DATA = "Partial Data"
    INSUFFICIENT_DATA = "Insufficient Data"


# ---------------------------------------------------------------------------
# Fundamentals Analyst
# ---------------------------------------------------------------------------


class FundamentalsCoverageBand(StrEnum):
    """Deterministic evidence-availability band for the Fundamentals Analyst."""

    SUFFICIENT_DATA = "Sufficient Data"
    PARTIAL_DATA = "Partial Data"
    INSUFFICIENT_DATA = "Insufficient Data"
