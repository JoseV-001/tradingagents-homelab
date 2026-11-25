"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    STRICT_PRICE_AND_EVIDENCE_RULES,
    build_evidence_coverage_summary,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        market_report = (state.get("market_report") or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)
        coverage_summary = build_evidence_coverage_summary(state)
        coverage_block = f"{coverage_summary}\n\n" if coverage_summary else ""

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's verified snapshot figures ONLY when executing an active Buy/Sell "
                "and ONLY via an explicit arithmetic calculation from verified numbers (such as exact "
                "Close and ATR shown step-by-step in Reasoning). Never invent arbitrary supports, "
                "resistances, targets, or stop-losses; if Action is Hold or no explicit formula "
                "validates a level, omit entry_price and stop_loss (leave null) and state explicitly "
                "that there is no validated technical price level. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing verified evidence to propose a disciplined transaction. "
                    "Based on the verified reports and research plan, provide a specific action: Buy, Sell, or Hold. "
                    + grounding
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; omit entry_price and stop_loss (null) whenever Action is Hold "
                    "or whenever no explicit mathematical derivation from verified snapshot data "
                    "exists, and never invent arbitrary levels. "
                    + STRICT_PRICE_AND_EVIDENCE_RULES
                    + " "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{coverage_block}"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strictly evidence-traceable trading proposal.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, strictly traceable to verified figures and preserving any Insufficient Data uncertainties (explicitly state if no validated technical price level exists)\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: omit Entry Price and Stop Loss (`not provided` / null) unless derived by explicit arithmetic from verified snapshot numbers"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
