from tradingagents.agents.context import (
    STRICT_PRICE_AND_EVIDENCE_RULES,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = opponent_argument_or_opening(
            risk_debate_state.get("current_aggressive_response", ""), "aggressive analyst"
        )
        current_conservative_response = opponent_argument_or_opening(
            risk_debate_state.get("current_conservative_response", ""), "conservative analyst"
        )

        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, weigh upside signals, downside risks, and data completeness objectively when evaluating the trader's proposal, adhering strictly to verified evidence.

{STRICT_PRICE_AND_EVIDENCE_RULES}

Here is the trader's decision:
{trader_decision}

Sources available:
{instrument_context}
{portfolio_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Current conversation history: {history}
Last response from the aggressive analyst: {current_aggressive_response}
Last response from the conservative analyst: {current_conservative_response}

If there are no responses from the other viewpoints yet, open with your own balanced assessment. Critique both over-optimism and excessive pessimism using ONLY verified data from the reports above, explicitly factor any `Insufficient Data` (`DATA_UNAVAILABLE`) gaps into position sizing, and NEVER invent company/macro facts or arbitrary price targets, supports, resistances, or stop-losses (state explicitly when no validated technical level exists). Output conversationally without special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
