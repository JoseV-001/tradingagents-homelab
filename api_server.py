"""Lightweight FastAPI bridge exposing TradingAgents chat and multi-agent analysis."""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Load environment variables from project .env if present (never log or expose them).
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

from tradingagents.dataflows.config import set_config  # noqa: E402
from tradingagents.default_config import build_default_config  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.llm_clients.google_client import GoogleClient  # noqa: E402

logger = logging.getLogger("tradingagents.api")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = "gemini-3.1-flash-lite"

app = FastAPI(
    title="TradingAgents API",
    version="1.0.0",
    description="FastAPI gateway for personal portfolio AI chat and multi-agent ticker analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str = Field(default="user")
    content: str = Field(default="")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    portfolio_context: dict[str, Any] | str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    date: str | None = None
    analysts: list[str] | None = None
    fast: bool = False


def _resolved_runtime_config() -> dict[str, Any]:
    cfg = build_default_config()
    cfg.setdefault("output_language", "Portuguese")
    tool_vendors = dict(cfg.get("tool_vendors") or {})
    tool_vendors.setdefault("get_fundamentals", "brapi,yfinance")
    cfg["tool_vendors"] = tool_vendors
    return cfg


def _resolved_model_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("quick_think_llm") or DEFAULT_MODEL)


def _build_chat_llm(cfg: dict[str, Any]):
    model_name = _resolved_model_name(cfg)
    client = GoogleClient(
        model=model_name,
        base_url=cfg.get("quick_think_backend_url") or cfg.get("backend_url"),
        thinking_level=cfg.get("google_thinking_level"),
        temperature=cfg.get("temperature"),
    )
    return client.get_llm(), model_name


def _format_portfolio_context(ctx: dict[str, Any] | str | None) -> str:
    if ctx is None:
        return "Contexto da carteira não fornecido."
    if isinstance(ctx, str):
        return ctx.strip() or "Contexto da carteira vazio."
    return json.dumps(ctx, ensure_ascii=False, indent=2)


@app.get("/")
def root():
    return health()


@app.get("/health")
def health():
    cfg = _resolved_runtime_config()
    return {
        "status": "ok",
        "model": _resolved_model_name(cfg),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY não configurada no servidor.")

    cfg = _resolved_runtime_config()
    llm, model_name = _build_chat_llm(cfg)
    portfolio_json = _format_portfolio_context(request.portfolio_context)

    system_prompt = (
        "Você é um Assessor Financeiro Sênior e Copiloto de Investimentos integrado ao sistema InvestPortfolio.\n"
        "O usuário possui investimentos multi-mercado na B3 (Brasil, em BRL) e nos EUA via Nomad (NYSE/NASDAQ, em USD).\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Baseie suas respostas EXCLUSIVAMENTE nos dados reais da carteira fornecidos abaixo. Nunca invente ativos, quantidades, cotações, dividendos ou saldos ausentes.\n"
        "2. Quando analisar saúde, diversificação, concentração, exposição cambial (USD/BRL) ou rentabilidade, cite exatamente os tickers, percentuais e valores observados no snapshot.\n"
        "3. Se faltar alguma informação no snapshot (ex.: histórico macro externo ou dados fundamentalistas completos), informe essa limitação claramente em vez de preencher por suposição.\n"
        "4. Responda sempre em português brasileiro claro, estruturado em Markdown elegante (listas, negritos, pequenas tabelas quando útil) e objetivo.\n\n"
        f"DADOS ATUAIS CONSOLIDADOS DA CARTEIRA:\n```json\n{portfolio_json}\n```"
    )

    messages = [SystemMessage(content=system_prompt)]
    for turn in request.history[-12:]:
        content = (turn.content or "").strip()
        if not content:
            continue
        if turn.role.lower() in ("assistant", "model", "ai"):
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=request.message.strip()))

    try:
        ai_message = await run_in_threadpool(llm.invoke, messages)
        reply = getattr(ai_message, "content", str(ai_message))
        if isinstance(reply, list):
            reply = "\n".join(str(item.get("text", item) if isinstance(item, dict) else item) for item in reply)
        return ChatResponse(reply=str(reply).strip(), model=model_name)
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=502, detail=f"Erro ao consultar modelo Gemini: {type(exc).__name__}: {exc}")


def _run_multi_agent_analysis(
    ticker: str,
    trade_date: str,
    selected_analysts: list[str],
    fast: bool,
) -> dict[str, Any]:
    cfg = _resolved_runtime_config()
    cfg["checkpoint_enabled"] = False
    if fast:
        cfg["max_debate_rounds"] = 1
        cfg["max_risk_discuss_rounds"] = 1
    set_config(cfg)

    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=False,
        config=cfg,
    )
    final_state, final_rating = graph.propagate(ticker, trade_date)

    inv_debate = final_state.get("investment_debate_state") or {}
    risk_debate = final_state.get("risk_debate_state") or {}

    return {
        "ticker": ticker,
        "date": trade_date,
        "analysts": selected_analysts,
        "model": _resolved_model_name(cfg),
        "final_rating": final_rating,
        "final_trade_decision": final_state.get("final_trade_decision", ""),
        "summary": {
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_plan": final_state.get("investment_plan", ""),
            "trader_investment_plan": final_state.get("trader_investment_plan", ""),
            "bull_history": inv_debate.get("bull_history", ""),
            "bear_history": inv_debate.get("bear_history", ""),
            "risk_history": risk_debate.get("history", ""),
        },
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY não configurada no servidor.")

    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker inválido.")

    trade_date = (request.date or date.today().strftime("%Y-%m-%d")).strip()
    valid_analysts = {"market", "social", "news", "fundamentals"}
    default_analysts = ["market", "news", "fundamentals"] if request.fast else ["market", "social", "news", "fundamentals"]
    selected = [a.strip().lower() for a in (request.analysts or default_analysts) if a.strip().lower() in valid_analysts]
    if not selected:
        selected = default_analysts

    try:
        return await run_in_threadpool(_run_multi_agent_analysis, ticker, trade_date, selected, request.fast)
    except Exception as exc:
        logger.exception("Multi-agent analysis failed for %s", ticker)
        raise HTTPException(status_code=502, detail=f"Erro na análise multi-agente de {ticker}: {type(exc).__name__}: {exc}")
