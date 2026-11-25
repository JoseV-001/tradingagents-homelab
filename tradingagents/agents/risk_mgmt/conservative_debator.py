from tradingagents.agents.context import (
    STRICT_PRICE_AND_EVIDENCE_RULES,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = opponent_argument_or_opening(
            risk_debate_state.get("current_aggressive_response", ""), "aggressive analyst"
        )
        current_neutral_response = opponent_argument_or_opening(
            risk_debate_state.get("current_neutral_response", ""), "neutral analyst"
        )

        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Conservative Risk Analyst, prioritize capital preservation, downside control, and epistemic discipline when evaluating the trader's proposal, adhering strictly to verified evidence.

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
Last response from the neutral analyst: {current_neutral_response}

If there are no responses from the other viewpoints yet, open with your own evidence-grounded risk assessment. Critique any unverified assumptions or blind-spot exposure (especially where analysts reported `Insufficient Data` / `DATA_UNAVAILABLE`) using ONLY verified data from the reports above â€” without inventing macro threats, company weaknesses, or arbitrary price targets, supports, resistances, or stop-losses (state explicitly when no validated technical level exists). Output conversationally without special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
