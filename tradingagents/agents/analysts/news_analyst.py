"""News analyst with deterministic tool-evidence auditing and anti-hallucination guards."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import WRAP_UP, take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.schemas import NewsCoverageBand
from tradingagents.agents.tools import (
    get_global_news,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)

_SOURCE_META = {
    "get_news": {
        "category": "Notícias Específicas do Ativo",
        "source_label": "Ticker News (get_news)",
    },
    "get_global_news": {
        "category": "Notícias Globais e Macroeconômicas",
        "source_label": "Global News (get_global_news)",
    },
    "get_macro_indicators": {
        "category": "Indicadores Macroeconômicos (FRED)",
        "source_label": "FRED Macro (get_macro_indicators)",
    },
    "get_prediction_markets": {
        "category": "Mercados de Previsão (Polymarket)",
        "source_label": "Polymarket (get_prediction_markets)",
    },
}

_COMMON_NO_DATA_MARKERS = (
    "data_unavailable:",
    "no_data_available:",
    "is not a date; give it as yyyy-mm-dd",
    "is after end_date",
)

_NO_DATA_MARKERS = {
    "get_news": _COMMON_NO_DATA_MARKERS + (
        "<yahoo finance news unavailable",
        "no news found for ",
        "nenhuma notícia disponível",
        "nenhuma noticia disponivel",
        "no news available",
        "no articles found",
        "no relevant news",
        "news unavailable",
        "no results found",
        "error retrieving news",
        "failed to fetch news",
    ),
    "get_global_news": _COMMON_NO_DATA_MARKERS + (
        "<yahoo finance global news unavailable",
        "no global news found between ",
        "no global news",
        "global news unavailable",
        "nenhuma notícia global",
    ),
    "get_macro_indicators": _COMMON_NO_DATA_MARKERS + (
        "is not a known macro alias or a valid fred series id",
        "not found. pass a known alias",
        "no observations for ",
        "fred request failed:",
        "fred_api_key environment variable is not set",
    ),
    "get_prediction_markets": _COMMON_NO_DATA_MARKERS + (
        "prediction-market odds are withheld for ",
        "polymarket data is currently unavailable",
        "no open prediction markets matched ",
    ),
}

_RETURNED_HEADER_RE = re.compile(r"^\[([a-zA-Z0-9_]+) returned\]\n(.*)$", re.DOTALL)
_CONFIDENCE_LINE_RE = re.compile(r"(\*\*Confidence:\*\*\s*)(?:\*\*)?(?:High|Medium|Alta|Média)(?:\*\*)?", re.IGNORECASE)


def _seven_days_back(trade_date: str) -> str:
    try:
        return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    except ValueError:
        return trade_date


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _is_empty_alpha_vantage_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    if "Error Message" in payload:
        return True
    if ("Information" in payload or "Note" in payload) and not payload.get("feed"):
        return True
    if "feed" in payload and not payload["feed"]:
        return True
    return str(payload.get("items", "")).strip() == "0"


def _has_substantive_data(block: str, tool_name: str) -> bool:
    """Detect empty, withheld, coverage-gapped, or failed tool outputs deterministically."""
    if tool_name not in _NO_DATA_MARKERS:
        raise ValueError(f"Ferramenta desconhecida no News Analyst: {tool_name}")

    raw = str(block or "").strip()
    if not raw:
        return False

    lowered = raw.lower()
    if any(marker in lowered for marker in _NO_DATA_MARKERS[tool_name]):
        return False

    if lowered.startswith("<") and any(
        marker in lowered[:350]
        for marker in ("unavailable", "no news", "fetch failed", "coverage gap")
    ):
        return False

    return not (tool_name in ("get_news", "get_global_news") and _is_empty_alpha_vantage_payload(raw))


def _summarize_unavailability(block: str) -> str:
    """Extract a concise factual diagnostic reason from an unavailable/withheld tool output."""
    raw = " ".join(str(block or "").strip().split())
    if not raw:
        return "DATA_UNAVAILABLE: retorno vazio da ferramenta."
    if raw.startswith("<") and raw.endswith(">"):
        return f"DATA_UNAVAILABLE ({raw[1:-1]})"
    for boiler in (
        " This says nothing about the instrument;",
        " Proceed without it;",
        " Do not estimate or fabricate values",
    ):
        idx = raw.find(boiler)
        if idx != -1:
            raw = raw[:idx].strip()
    return raw[:280]


def _audit_tool_results(messages: list) -> dict[str, dict]:
    """Inspect message history and classify each News Analyst tool deterministically."""
    audit = {
        name: {
            "category": meta["category"],
            "source_label": meta["source_label"],
            "called": False,
            "has_data": False,
            "reasons": [],
        }
        for name, meta in _SOURCE_META.items()
    }

    call_id_to_name: dict[str, str] = {}
    for msg in messages or ():
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                cid = call.get("id")
                cname = call.get("name")
                if cid and cname:
                    call_id_to_name[cid] = cname
        elif isinstance(msg, ToolMessage):
            tname = msg.name or call_id_to_name.get(getattr(msg, "tool_call_id", ""), "")
            if tname not in audit:
                continue
            text = _content_to_text(msg.content)
            entry = audit[tname]
            entry["called"] = True
            if _has_substantive_data(text, tname):
                entry["has_data"] = True
            else:
                reason = _summarize_unavailability(text)
                if reason not in entry["reasons"]:
                    entry["reasons"].append(reason)
        elif isinstance(msg, HumanMessage):
            text = _content_to_text(msg.content)
            match = _RETURNED_HEADER_RE.match(text.strip())
            if match:
                tname, body = match.group(1), match.group(2)
                if tname in audit:
                    entry = audit[tname]
                    entry["called"] = True
                    if _has_substantive_data(body, tname):
                        entry["has_data"] = True
                    else:
                        reason = _summarize_unavailability(body)
                        if reason not in entry["reasons"]:
                            entry["reasons"].append(reason)

    return audit


def _format_audit_block(ticker: str, audit: dict[str, dict]) -> str:
    """Build deterministic Python source-of-truth block for Turn 2+ prompts."""
    ticker_entry = audit["get_news"]
    if ticker_entry["has_data"]:
        ticker_status = "AVAILABLE (artigos contemporâneos verificados)"
    elif ticker_entry["called"]:
        ticker_status = f"{NewsCoverageBand.INSUFFICIENT_DATA.value} (DATA_UNAVAILABLE)"
    else:
        ticker_status = "NOT_YET_QUERIED"

    any_other = any(
        audit[k]["has_data"]
        for k in ("get_global_news", "get_macro_indicators", "get_prediction_markets")
    )
    if ticker_entry["has_data"]:
        overall_band = NewsCoverageBand.SUFFICIENT_DATA.value
    elif any_other:
        overall_band = NewsCoverageBand.PARTIAL_DATA.value
    else:
        overall_band = NewsCoverageBand.INSUFFICIENT_DATA.value

    lines = [
        "",
        "VERIFIED TOOL EVIDENCE AUDIT — PYTHON SOURCE OF TRUTH:",
        f"- Classificação Determinística Geral: {overall_band}",
        f"- Cobertura Específica de `{ticker}` (`get_news`): {ticker_status}",
    ]
    for tname, entry in audit.items():
        if not entry["called"]:
            state_str = "NÃO CONSULTADA AINDA"
        elif entry["has_data"]:
            state_str = "DADOS VERIFICÁVEIS DISPONÍVEIS"
        else:
            detail = "; ".join(entry["reasons"]) or "DATA_UNAVAILABLE"
            state_str = f"INDISPONÍVEL / SEM EVIDÊNCIAS ({detail}) — NÃO chame `{tname}` novamente"
        lines.append(f"- `{tname}` ({entry['category']}): {state_str}")

    if ticker_entry["called"] and not ticker_entry["has_data"]:
        lines.append(
            f"- REGRA CRÍTICA ATIVA: Não há notícias verificáveis para `{ticker}`. "
            f"Use `**Confidence:** Low`, declare explicitamente `Insufficient Data` (`DATA_UNAVAILABLE`) "
            f"na seção do ativo e NÃO escreva descrição genérica da empresa, holdings, dividendos ou setor por memória."
        )
    return "\n".join(lines)


def _build_insufficient_data_report(
    ticker: str,
    start_date: str,
    end_date: str,
    audit: dict[str, dict],
) -> str:
    """Deterministic report when no tool returned usable contemporary evidence."""
    def _reason_for(tname: str) -> str:
        entry = audit[tname]
        if not entry["called"]:
            return "DATA_UNAVAILABLE: ferramenta não retornou dados verificáveis nesta execução."
        return "; ".join(entry["reasons"]) or "DATA_UNAVAILABLE: sem dados contemporâneos verificáveis."

    news_reason = _reason_for("get_news")
    global_reason = _reason_for("get_global_news")
    macro_reason = _reason_for("get_macro_indicators")
    poly_reason = _reason_for("get_prediction_markets")

    return "\n".join([
        f"- **Overall News Status:** **{NewsCoverageBand.INSUFFICIENT_DATA.value}** (`DATA_UNAVAILABLE`)",
        f"- **Ticker News Coverage ({ticker}):** **{NewsCoverageBand.INSUFFICIENT_DATA.value}** (`DATA_UNAVAILABLE`)",
        "- **Confidence:** Low",
        "",
        f"Não foi possível produzir análise de notícias ou cenário macroeconômico para `{ticker}` "
        f"na janela de `{start_date}` a `{end_date}` porque nenhuma das fontes consultadas retornou "
        "evidências contemporâneas verificáveis. Conforme protocolo anti-alucinação, nenhum perfil "
        "corporativo genérico ou narrativa macroeconômica baseada em memória do modelo foi incluído.",
        "",
        f"### 1. Notícias Específicas do Ativo (`{ticker}`)",
        f"- **Status:** `{NewsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {news_reason}",
        "- Nenhuma manchete ou evento corporativo contemporâneo pôde ser verificado para o ativo na janela analisada.",
        "",
        "### 2. Notícias Globais e Macroeconômicas",
        f"- **Status:** `{NewsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {global_reason}",
        "",
        "### 3. Indicadores Macroeconômicos (FRED)",
        f"- **Status:** `{NewsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {macro_reason}",
        "",
        "### 4. Mercados de Previsão (Polymarket)",
        f"- **Status:** `{NewsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {poly_reason}",
        "",
        "### 5. Falhas e Indisponibilidade de Fontes",
        f"- `get_news` ({audit['get_news']['category']}): {news_reason}",
        f"- `get_global_news` ({audit['get_global_news']['category']}): {global_reason}",
        f"- `get_macro_indicators` ({audit['get_macro_indicators']['category']}): {macro_reason}",
        f"- `get_prediction_markets` ({audit['get_prediction_markets']['category']}): {poly_reason}",
        "- Indisponibilidade de fornecedor, lacuna de cobertura histórica ou retenção *point-in-time* "
        "não equivalem a ausência de fatos relevantes nem a cenário neutro; representam ausência de evidência verificável.",
        "",
        "### 6. Tabela Resumo de Evidências",
        "",
        "| Categoria | Ferramenta | Status | Evidência / Motivo da Indisponibilidade |",
        "| --- | --- | --- | --- |",
        f"| Notícias Específicas (`{ticker}`) | `get_news` | `DATA_UNAVAILABLE` (`Insufficient Data`) | {news_reason} |",
        f"| Notícias Globais | `get_global_news` | `DATA_UNAVAILABLE` (`Insufficient Data`) | {global_reason} |",
        f"| Indicadores Macro (FRED) | `get_macro_indicators` | `DATA_UNAVAILABLE` (`Insufficient Data`) | {macro_reason} |",
        f"| Mercados de Previsão (Polymarket) | `get_prediction_markets` | `DATA_UNAVAILABLE` (`Insufficient Data`) | {poly_reason} |",
    ])


def _ensure_partial_ticker_header(ticker: str, report: str) -> str:
    """Guarantee explicit Insufficient Data / DATA_UNAVAILABLE header and Low confidence when ticker news failed."""
    normalized = _CONFIDENCE_LINE_RE.sub(r"\1Low", report)
    lowered = normalized.lower()
    if "insufficient data" in lowered and "data_unavailable" in lowered:
        return normalized
    header = "\n".join([
        f"- **Overall News Status:** **{NewsCoverageBand.PARTIAL_DATA.value}** "
        f"(Ticker `{ticker}`: **{NewsCoverageBand.INSUFFICIENT_DATA.value}** / `DATA_UNAVAILABLE`)",
        f"- **Ticker News Coverage ({ticker}):** **{NewsCoverageBand.INSUFFICIENT_DATA.value}** (`DATA_UNAVAILABLE`)",
        "- **Confidence:** Low",
        "",
    ])
    return header + normalized.lstrip()


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        start_date = _seven_days_back(current_date)
        ticker = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        messages = state.get("messages", [])

        audit = _audit_tool_results(messages)
        any_called = any(entry["called"] for entry in audit.values())
        all_called = all(entry["called"] for entry in audit.values())
        any_substantive = any(entry["has_data"] for entry in audit.values())
        is_wrap_up = bool(
            messages
            and isinstance(messages[-1], HumanMessage)
            and messages[-1].content == WRAP_UP
        )

        # Curto-circuito determinístico: se todas as categorias já foram consultadas
        # (ou a rodada final wrap_up chegou) e NENHUMA retornou dados verificáveis,
        # não invocamos o LLM para impedir alucinação com memória de treinamento.
        if any_called and not any_substantive and (all_called or is_wrap_up):
            fallback_report = _build_insufficient_data_report(
                ticker=ticker,
                start_date=start_date,
                end_date=current_date,
                audit=audit,
            )
            return {
                "messages": [AIMessage(content=fallback_report)],
                "news_report": fallback_report,
            }

        audit_section = _format_audit_block(ticker, audit) if any_called else ""

        system_message = (
            f"Você é um pesquisador sênior de notícias financeiras e macroeconomia encarregado de analisar "
            f"exclusivamente evidências contemporâneas verificáveis na janela de {start_date} a {current_date} "
            f"para o {asset_label} `{ticker}`.\n\n"
            f"FERRAMENTAS DISPONÍVEIS E PROTOCOLO DE COLETA:\n"
            f"1. `get_news(start_date, end_date)` — notícias específicas sobre o {asset_label} sob análise (ex.: `{start_date}` a `{current_date}`).\n"
            f"2. `get_global_news(curr_date, look_back_days, limit)` — manchetes globais e macroeconômicas na janela.\n"
            f"3. `get_macro_indicators(indicator, curr_date, look_back_days)` — séries macroeconômicas oficiais do FRED "
            f"(ex.: `'fed_funds_rate'`, `'cpi'`, `'core_pce'`, `'unemployment'`, `'10y_treasury'`, `'yield_curve'`).\n"
            f"4. `get_prediction_markets(topic, limit)` — probabilidades implícitas de eventos futuros via Polymarket "
            f"(ex.: `'Fed rate cut'`, `'recession 2026'`).\n"
            f"- Na primeira rodada de ferramentas, chame as 4 ferramentas em conjunto para verificar todas as fontes de uma só vez.\n"
            f"- NUNCA repita chamadas para uma ferramenta que já tenha retornado `DATA_UNAVAILABLE`, `NO_DATA_AVAILABLE`, "
            f"`<... unavailable ...>` ou retenção *point-in-time* (`withheld`).\n\n"
            f"REGRAS OBRIGATÓRIAS DE EVIDÊNCIA E ANTI-ALUCINAÇÃO:\n"
            f"1. Utilize EXCLUSIVAMENTE fatos, títulos, fontes, datas, valores e probabilidades presentes literalmente "
            f"nos retornos das ferramentas desta execução.\n"
            f"2. PROIBIÇÃO DE TEXTO GENÉRICO SOBRE A EMPRESA: Se `get_news(start_date, end_date)` não retornar notícias "
            f"contemporâneas verificáveis para `{ticker}` (ex.: `DATA_UNAVAILABLE`, lacuna de cobertura ou zero artigos), "
            f"registre explicitamente `Insufficient Data` (`DATA_UNAVAILABLE`) para o ativo e defina `Confidence` como `Low`. "
            f"É ESTRITAMENTE PROIBIDO preencher o relatório com descrição institucional da empresa, histórico societário, "
            f"holdings/controladas, perfil tradicional de dividendos ou comentários setoriais baseados na sua memória de treinamento.\n"
            f"3. PROIBIÇÃO DE CENÁRIO MACRO INVENTADO: Nunca cite taxas de juros (Selic, Fed Funds, BCE), inflação "
            f"(IPCA, CPI, PCE), PIB, quadro fiscal, decisões de bancos centrais ou eventos geopolíticos que não "
            f"constem explicitamente nos retornos de `get_global_news`, `get_macro_indicators` ou `get_prediction_markets`.\n"
            f"4. NÃO EXTRAPOLE ESCOPO: Não trate manchetes globais ou norte-americanas como se fossem notícias específicas "
            f"de `{ticker}`, salvo se o artigo retornado mencionar `{ticker}` nominalmente.\n"
            f"5. FALHA NÃO É NEUTRALIDADE: Fontes indisponíveis, sem chave configurada ou retidas por data histórica "
            f"devem ser reportadas como `DATA_UNAVAILABLE` / `Insufficient Data`, nunca como cenário neutro ou calmo.\n"
            f"6. Não faça recomendações de compra ou venda nem invente preços-alvo, suportes ou resistências.\n\n"
            f"ESTRUTURA OBRIGATÓRIA DO RELATÓRIO:\n"
            f"Inicie com os três itens de cabeçalho em lista:\n"
            f"- **Overall News Status:** **Sufficient Data** | **Partial Data** | **Insufficient Data**\n"
            f"- **Ticker News Coverage ({ticker}):** **Available** | **Insufficient Data (DATA_UNAVAILABLE)**\n"
            f"- **Confidence:** High | Medium | Low (obrigatoriamente `Low` quando `get_news` estiver indisponível)\n\n"
            f"Em seguida, separe obrigatoriamente nas 6 seções abaixo:\n"
            f"### 1. Notícias Específicas do Ativo (`{ticker}`)\n"
            f"### 2. Notícias Globais e Macroeconômicas\n"
            f"### 3. Indicadores Macroeconômicos (FRED)\n"
            f"### 4. Mercados de Previsão (Polymarket)\n"
            f"### 5. Falhas e Indisponibilidade de Fontes\n"
            f"### 6. Tabela Resumo de Evidências\n"
            f"Inclua ao final uma tabela Markdown com as colunas `| Categoria | Fonte / Ferramenta | Status | Evidência Verificada |`.\n"
            + audit_section
            + get_language_instruction()
            + "\n(Nota: este relatório analítico NÃO possui linha de Rating "
            + "nem proposta final de transação; encerre imediatamente após a tabela resumo.)"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " Report strictly what your tools return; never fill missing tool data from general knowledge."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, messages)

        # Guarda pós-turno: se o modelo encerrou sem novas tool_calls e todas as
        # ferramentas consultadas falharam/retornaram vazio, garantimos o relatório
        # determinístico de indisponibilidade sem texto livre potencialmente alucinado.
        if not getattr(result, "tool_calls", None) and any_called:
            if not any_substantive:
                report = _build_insufficient_data_report(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=current_date,
                    audit=audit,
                )
                result = AIMessage(content=report)
            elif audit["get_news"]["called"] and not audit["get_news"]["has_data"] and report:
                report = _ensure_partial_ticker_header(ticker, report)
                result = AIMessage(content=report)

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
