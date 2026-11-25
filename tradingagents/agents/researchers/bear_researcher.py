from tradingagents.agents.context import (
    STRICT_PRICE_AND_EVIDENCE_RULES,
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bull analyst"
        )
        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        prompt = f"""You are a Bear Analyst making the cautious/risk case on the {target_label}, strictly grounded in verified analyst evidence.

{STRICT_PRICE_AND_EVIDENCE_RULES}

Key points to focus on (ONLY where supported by the reports below):
- Verified Adverse or Limiting Signals: Highlight exact adverse or diverging indicators actually present in the reports (for example, close below SMA 50, negative MACD below zero, mixed trend classification, or verified fundamental headwinds when available).
- Epistemic Risk from Missing Data: When fundamentals, news, or sentiment report `Insufficient Data` (`DATA_UNAVAILABLE`), highlight the risk of committing capital under blind spots â€” WITHOUT inventing macroeconomic crises, sector headwinds, competitive weaknesses, or financial numbers from training memory.
- Bull Counterpoints: Critically challenge any over-optimistic interpretation by the bull analyst using strictly the verified figures in the reports.
- Engagement: Present a clear, conversational critique grounded 100% in the reported evidence, explicitly stating when no validated technical support, resistance, stop-loss, or downside target exists.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use ONLY this verified information to present the risk case without fabricating facts or price levels.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
