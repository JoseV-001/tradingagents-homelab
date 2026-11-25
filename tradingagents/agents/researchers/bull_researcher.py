from tradingagents.agents.context import (
    STRICT_PRICE_AND_EVIDENCE_RULES,
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bear analyst"
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

        prompt = f"""You are a Bull Analyst advocating for the constructive case on the {target_label}, strictly grounded in verified analyst evidence.

{STRICT_PRICE_AND_EVIDENCE_RULES}

Key points to focus on (ONLY where supported by the reports below):
- Verified Constructive Signals: Highlight exact positive technical, fundamental, sentiment, or news facts actually present in the reports (for example, close above specific moving averages, positive MACD histogram, or verified financial metrics when available).
- Epistemic Honesty: If fundamentals, news, or sentiment report `Insufficient Data` (`DATA_UNAVAILABLE`), acknowledge clearly that those dimensions are unavailable rather than inventing revenue projections, dividends, holdings, or macro tailwinds.
- Bear Counterpoints: Critically examine the bear analyst's argument, pointing out any over-interpretation of mixed signals or any unverified assumptions, using strictly the verified numbers in the reports.
- Engagement: Present a clear, conversational rebuttal grounded 100% in the reported evidence, explicitly stating when no validated technical support, resistance, or price target exists.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use ONLY this verified information to present the constructive case without fabricating facts or price levels.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
