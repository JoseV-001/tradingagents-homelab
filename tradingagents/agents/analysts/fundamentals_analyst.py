"""Fundamentals analyst with deterministic tool-evidence auditing and anti-hallucination guards."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import WRAP_UP, take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.schemas import FundamentalsCoverageBand
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions,
)

_CORE_FINANCIAL_TOOLS = (
    "get_fundamentals",
    "get_balance_sheet",
    "get_income_statement",
    "get_cashflow",
)

_SOURCE_META = {
    "get_fundamentals": {
        "category": "Visão Geral e Múltiplos",
        "source_label": "Overview / Múltiplos (get_fundamentals)",
    },
    "get_balance_sheet": {
        "category": "Balanço Patrimonial",
        "source_label": "Balance Sheet (get_balance_sheet)",
    },
    "get_income_statement": {
        "category": "Demonstração de Resultados (DRE)",
        "source_label": "Income Statement (get_income_statement)",
    },
    "get_cashflow": {
        "category": "Demonstração de Fluxo de Caixa (DFC)",
        "source_label": "Cash Flow (get_cashflow)",
    },
    "get_insider_transactions": {
        "category": "Transações de Insiders",
        "source_label": "Insider Transactions (get_insider_transactions)",
    },
}

_COMMON_NO_DATA_MARKERS = (
    "data_unavailable:",
    "no_data_available:",
    "are withheld for this date",
    "is withheld for this date",
)

_NO_DATA_MARKERS = {
    "get_fundamentals": _COMMON_NO_DATA_MARKERS + (
        "profile fundamentals are withheld for this date",
        "no usable market data",
        "no fundamental data",
    ),
    "get_balance_sheet": _COMMON_NO_DATA_MARKERS + (
        "balance sheet data is withheld for this date",
        "no balance sheet data",
    ),
    "get_income_statement": _COMMON_NO_DATA_MARKERS + (
        "income statement data is withheld for this date",
        "no income statement data",
    ),
    "get_cashflow": _COMMON_NO_DATA_MARKERS + (
        "cash flow data is withheld for this date",
        "no cash flow data",
    ),
    "get_insider_transactions": _COMMON_NO_DATA_MARKERS + (
        "insider transactions are withheld for this date",
        "no insider transactions reported for symbol",
        "no insider transactions",
    ),
}

_RETURNED_HEADER_RE = re.compile(r"^\[([a-zA-Z0-9_]+) returned\]\n(.*)$", re.DOTALL)
_CONFIDENCE_LINE_RE = re.compile(
    r"(\*\*Confidence:\*\*\s*)(?:\*\*)?(?:High|Medium|Alta|Média)(?:\*\*)?",
    re.IGNORECASE,
)


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
    if not isinstance(payload, dict) or not payload:
        return True
    if "Error Message" in payload:
        return True
    if "Information" in payload or "Note" in payload:
        has_records = any(
            bool(payload.get(k))
            for k in ("Symbol", "annualReports", "quarterlyReports", "data")
        )
        if not has_records:
            return True
    if "annualReports" in payload and "quarterlyReports" in payload:
        if not payload["annualReports"] and not payload["quarterlyReports"]:
            return True
    if "data" in payload and payload["data"] == []:
        return True
    return False


def _detect_vendor_label(text: str) -> str:
    raw = str(text or "")
    if "(Brapi)" in raw:
        return "Brapi (B3)"
    if "SEC EDGAR facts filed on or before" in raw:
        return "SEC EDGAR"
    if raw.startswith("# Company Fundamentals for ") or raw.startswith("# Balance Sheet data for ") \
            or raw.startswith("# Cash Flow data for ") or raw.startswith("# Income Statement data for ") \
            or raw.startswith("# Insider Transactions data for "):
        return "Yahoo Finance (yfinance)"
    if raw.strip().startswith("{"):
        return "Alpha Vantage"
    return "Vendor configurado"


def _has_substantive_data(block: str, tool_name: str) -> bool:
    """Detect empty, withheld, or failed fundamental tool outputs deterministically."""
    if tool_name not in _NO_DATA_MARKERS:
        raise ValueError(f"Ferramenta desconhecida no Fundamentals Analyst: {tool_name}")

    raw = str(block or "").strip()
    if not raw:
        return False

    lowered = raw.lower()
    if any(marker in lowered for marker in _NO_DATA_MARKERS[tool_name]):
        return False

    if _is_empty_alpha_vantage_payload(raw):
        return False

    return True


def _summarize_unavailability(block: str) -> str:
    """Extract concise factual reason from a withheld/unavailable fundamental tool output."""
    raw = str(block or "").strip()
    if not raw:
        return "DATA_UNAVAILABLE: retorno vazio da ferramenta."
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    compact = " ".join(lines) if lines else " ".join(raw.split())
    for boiler in (
        " This says nothing about the instrument;",
        " Do not estimate or fabricate values",
    ):
        idx = compact.find(boiler)
        if idx != -1:
            compact = compact[:idx].strip()
    if not compact.upper().startswith(("DATA_UNAVAILABLE", "NO_DATA_AVAILABLE")):
        return f"DATA_UNAVAILABLE ({compact[:260]})"
    return compact[:280]


def _audit_tool_results(messages: list) -> dict[str, dict]:
    """Inspect message history and classify each Fundamentals Analyst tool deterministically."""
    audit = {
        name: {
            "category": meta["category"],
            "source_label": meta["source_label"],
            "called": False,
            "has_data": False,
            "vendors": [],
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
                vendor = _detect_vendor_label(text)
                if vendor not in entry["vendors"]:
                    entry["vendors"].append(vendor)
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
                        vendor = _detect_vendor_label(body)
                        if vendor not in entry["vendors"]:
                            entry["vendors"].append(vendor)
                    else:
                        reason = _summarize_unavailability(body)
                        if reason not in entry["reasons"]:
                            entry["reasons"].append(reason)

    return audit


def _classify_coverage(audit: dict[str, dict]) -> tuple[FundamentalsCoverageBand, bool]:
    """Return (overall_band, has_core_financials)."""
    core_available = [audit[k]["has_data"] for k in _CORE_FINANCIAL_TOOLS]
    has_core = any(core_available)
    if all(core_available):
        return FundamentalsCoverageBand.SUFFICIENT_DATA, True
    if has_core or audit["get_insider_transactions"]["has_data"]:
        return FundamentalsCoverageBand.PARTIAL_DATA, has_core
    return FundamentalsCoverageBand.INSUFFICIENT_DATA, False


def _format_audit_block(ticker: str, audit: dict[str, dict]) -> str:
    """Build deterministic Python source-of-truth block for Turn 2+ prompts."""
    band, has_core = _classify_coverage(audit)
    lines = [
        "",
        "VERIFIED FUNDAMENTALS EVIDENCE AUDIT — PYTHON SOURCE OF TRUTH:",
        f"- Classificação Determinística de Cobertura: {band.value}",
        f"- Demonstrações / Múltiplos Financeiros Verificáveis (`{ticker}`): "
        f"{'AVAILABLE' if has_core else 'INSUFFICIENT DATA (DATA_UNAVAILABLE)'}",
    ]
    for tname, entry in audit.items():
        if not entry["called"]:
            state_str = "NÃO CONSULTADA AINDA"
        elif entry["has_data"]:
            v_str = ", ".join(entry["vendors"]) or "Vendor configurado"
            state_str = f"DADOS VERIFICÁVEIS DISPONÍVEIS (Fonte: {v_str})"
        else:
            detail = "; ".join(entry["reasons"]) or "DATA_UNAVAILABLE"
            state_str = (
                f"INDISPONÍVEL / RETIDA ({detail}) — NÃO chame `{tname}` novamente "
                f"(nem alterando `freq`, pois a restrição do fornecedor/data persiste)"
            )
        lines.append(f"- `{tname}` ({entry['category']}): {state_str}")

    if not has_core and all(audit[k]["called"] for k in _CORE_FINANCIAL_TOOLS):
        lines.append(
            f"- REGRA CRÍTICA ATIVA: Nenhuma demonstração contábil ou múltiplo verificável foi "
            f"retornado para `{ticker}`. Declare `Insufficient Data` (`DATA_UNAVAILABLE`), "
            f"use `Confidence: Low` e NÃO use conhecimento geral para citar lucro, receita, "
            f"P/L, P/VP, ROE, dívida, dividendos ou estrutura societária."
        )
    return "\n".join(lines)


def _build_insufficient_data_report(
    ticker: str,
    trade_date: str,
    audit: dict[str, dict],
) -> str:
    """Deterministic report when no valid financial statements or fundamental metrics exist."""
    def _reason_for(tname: str) -> str:
        entry = audit[tname]
        if not entry["called"]:
            return "DATA_UNAVAILABLE: ferramenta não retornou dados verificáveis nesta execução."
        return "; ".join(entry["reasons"]) or "DATA_UNAVAILABLE: sem dados fundamentalistas verificáveis."

    f_reason = _reason_for("get_fundamentals")
    bs_reason = _reason_for("get_balance_sheet")
    is_reason = _reason_for("get_income_statement")
    cf_reason = _reason_for("get_cashflow")
    in_status = (
        f"Disponível ({', '.join(audit['get_insider_transactions']['vendors'])})"
        if audit["get_insider_transactions"]["has_data"]
        else _reason_for("get_insider_transactions")
    )

    return "\n".join([
        f"- **Overall Fundamentals Status:** **{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}** (`DATA_UNAVAILABLE`)",
        f"- **Financial Statements Coverage ({ticker}):** **{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}** (`DATA_UNAVAILABLE`)",
        "- **Confidence:** Low",
        "",
        f"Não foi possível realizar análise fundamentalista para `{ticker}` na data-base `{trade_date}` "
        "porque as ferramentas configuradas não retornaram demonstrações contábeis ou indicadores "
        "fundamentalistas verificáveis (dados ausentes, sem cobertura SEC EDGAR para emissores fora dos EUA "
        "ou retidos por proteção *point-in-time* sem data de arquivamento conhecida). "
        "Conforme protocolo anti-alucinação, nenhum número financeiro, múltiplo, histórico de dividendos "
        "ou perfil societário foi extraído da memória geral do modelo.",
        "",
        f"### 1. Visão Geral e Múltiplos (`get_fundamentals`)",
        f"- **Status:** `{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {f_reason}",
        "",
        "### 2. Balanço Patrimonial (`get_balance_sheet`)",
        f"- **Status:** `{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {bs_reason}",
        "",
        "### 3. Demonstração de Resultados — DRE (`get_income_statement`)",
        f"- **Status:** `{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {is_reason}",
        "",
        "### 4. Demonstração de Fluxo de Caixa — DFC (`get_cashflow`)",
        f"- **Status:** `{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`).",
        f"- **Detalhe observado:** {cf_reason}",
        "",
        "### 5. Transações de Insiders (`get_insider_transactions`)",
        f"- **Status:** {'Disponível' if audit['get_insider_transactions']['has_data'] else f'`{FundamentalsCoverageBand.INSUFFICIENT_DATA.value}` (`DATA_UNAVAILABLE`)'}.",
        f"- **Detalhe observado:** {in_status}",
        "",
        "### 6. Tabela Resumo de Evidências Fundamentalistas",
        "",
        "| Categoria | Ferramenta | Período / Competência | Moeda / Unidade | Status / Fonte | Observação Verificada |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Visão Geral e Múltiplos | `get_fundamentals` | N/A | N/A | `DATA_UNAVAILABLE` | {f_reason} |",
        f"| Balanço Patrimonial | `get_balance_sheet` | N/A | N/A | `DATA_UNAVAILABLE` | {bs_reason} |",
        f"| DRE | `get_income_statement` | N/A | N/A | `DATA_UNAVAILABLE` | {is_reason} |",
        f"| Fluxo de Caixa | `get_cashflow` | N/A | N/A | `DATA_UNAVAILABLE` | {cf_reason} |",
        f"| Transações de Insiders | `get_insider_transactions` | N/A | N/A | "
        f"{'Disponível' if audit['get_insider_transactions']['has_data'] else '`DATA_UNAVAILABLE`'} | {in_status} |",
    ])


def _ensure_partial_header(ticker: str, report: str, band: FundamentalsCoverageBand) -> str:
    """Ensure partial fundamental coverage explicitly states missing sources and caps confidence if needed."""
    core_missing = band != FundamentalsCoverageBand.SUFFICIENT_DATA
    normalized = report
    if not core_missing:
        return normalized
    if "**Overall Fundamentals Status:**" in normalized:
        return normalized
    header = "\n".join([
        f"- **Overall Fundamentals Status:** **{band.value}**",
        f"- **Ticker:** `{ticker}`",
        "",
    ])
    return header + normalized.lstrip()


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        messages = state.get("messages", [])

        audit = _audit_tool_results(messages)
        any_called = any(entry["called"] for entry in audit.values())
        all_called = all(entry["called"] for entry in audit.values())
        all_core_called = all(audit[k]["called"] for k in _CORE_FINANCIAL_TOOLS)
        band, has_core = _classify_coverage(audit)
        is_wrap_up = bool(
            messages
            and isinstance(messages[-1], HumanMessage)
            and messages[-1].content == WRAP_UP
        )

        # Curto-circuito determinístico: se todas as ferramentas (ou todas as 4
        # ferramentas financeiras centrais em conjunto com insiders / wrap_up) foram
        # consultadas e NENHUMA demonstração ou múltiplo válido foi retornado,
        # emitimos o relatório determinístico de indisponibilidade sem chamar o LLM.
        if any_called and not has_core and (all_called or is_wrap_up):
            fallback_report = _build_insufficient_data_report(
                ticker=ticker,
                trade_date=current_date,
                audit=audit,
            )
            return {
                "messages": [AIMessage(content=fallback_report)],
                "fundamentals_report": fallback_report,
            }

        audit_section = _format_audit_block(ticker, audit) if any_called else ""

        system_message = (
            f"Você é um analista fundamentalista quantitativo e contábil responsável por auditar "
            f"as demonstrações financeiras e indicadores de `{ticker}` na data-base `{current_date}`.\n\n"
            f"FERRAMENTAS DISPONÍVEIS E PROTOCOLO DE COLETA:\n"
            f"1. `get_fundamentals(curr_date)` — visão geral, múltiplos e métricas TTM quando disponíveis.\n"
            f"2. `get_balance_sheet(freq, curr_date)` — balanço patrimonial (`quarterly` ou `annual`).\n"
            f"3. `get_income_statement(freq, curr_date)` — demonstração de resultados / DRE (`quarterly` ou `annual`).\n"
            f"4. `get_cashflow(freq, curr_date)` — demonstração de fluxo de caixa / DFC (`quarterly` ou `annual`).\n"
            f"5. `get_insider_transactions()` — movimentações societárias de insiders.\n"
            f"- Na primeira rodada de ferramentas, chame as 5 ferramentas em paralelo (`get_fundamentals`, "
            f"`get_balance_sheet`, `get_income_statement`, `get_cashflow` e `get_insider_transactions`) "
            f"para coletar todas as demonstrações de uma só vez.\n"
            f"- NUNCA repita chamadas para ferramentas que retornarem `DATA_UNAVAILABLE`, `NO_DATA_AVAILABLE`, "
            f"`withheld for this date` ou ausência de registros (inclusive trocando `quarterly` por `annual` quando "
            f"o retorno indicar retenção *point-in-time* por ausência de filing date).\n\n"
            f"REGRAS OBRIGATÓRIAS DE RIGOR CONTÁBIL E ANTI-ALUCINAÇÃO:\n"
            f"1. RASTREABILIDADE INTEGRAL DE CADA NÚMERO: Todo número financeiro citado no texto ou na tabela DEVE "
            f"vir acompanhado explicitamente de:\n"
            f"   (a) Valor exato observado;\n"
            f"   (b) Período / competência exata (ex.: `TTM em {current_date}`, `trimestre encerrado em 2026-09-30`, `exercício anual 2025-12-31`);\n"
            f"   (c) Moeda explícita retornada pela ferramenta (ex.: `BRL`, `USD`, `GBP`, `TWD` — se a ferramenta imprimir o número sem moeda explícita, escreva `moeda não informada pela fonte`, nunca adivinhe);\n"
            f"   (d) Unidade / escala exata (ex.: `unidades monetárias brutas` nas tabelas CSV do Yahoo Finance, `USD em milhões` no SEC EDGAR, `por ação`, `múltiplo (x)`, `fração decimal` vs `percentual (%)`);\n"
            f"   (e) Fonte e ferramenta exatas (ex.: `Yahoo Finance via get_fundamentals`, `Yahoo Finance via get_balance_sheet (quarterly)`, `SEC EDGAR via get_income_statement`).\n"
            f"2. ATENÇÃO ÀS ESCALAS DO YAHOO FINANCE:\n"
            f"   - `Dividend Yield` (ex.: `0.75%`) e `Debt to Equity` (ex.: `11.799% (0.12x)`) já vêm em percentual; NUNCA multiplique por 100 novamente.\n"
            f"   - `Profit Margin`, `Operating Margin`, `Return on Equity` e `Return on Assets` vêm em fração decimal (ex.: `0.18927` = `18,93%`; `2.1777` = `217,77%`); cite a fração original e, se converter para `%`, preserve a equivalência matemática exata.\n"
            f"   - Nas tabelas CSV do Yahoo Finance (`get_balance_sheet`, `get_income_statement`, `get_cashflow`), os valores estão em unidades inteiras brutas (ex.: `97258000000.0` = `97,258 bilhões`), diferentemente do SEC EDGAR que informa `USD in millions`.\n"
            f"3. PROIBIÇÃO ABSOLUTA DE MEMÓRIA GERAL DO LLM:\n"
            f"   - Se as ferramentas não retornarem demonstrações ou dados válidos, registre explicitamente `Insufficient Data` (`DATA_UNAVAILABLE`) e encerre sem simular análise.\n"
            f"   - É ESTRITAMENTE PROIBIDO usar memória de treinamento para mencionar controladas/participações societárias (ex.: explicar estrutura de holding ou subsidiárias não listadas no retorno), histórico de dividendos, guidance, eventos corporativos, market share ou comparações com médias setoriais/concorrentes não presentes nos retornos.\n"
            f"   - Se houver divergência contábil aparente nos dados brutos (ex.: `Net Income` superior a `Revenue (TTM)` em `get_fundamentals`), registre apenas o fato contábil observado nos números retornados sem inventar explicações societárias externas.\n"
            f"4. PROIBIÇÃO DE VALUATION INVENTADO E RECOMENDAÇÃO: Nunca invente preço-justo, preço-alvo, DCF, custo de capital (WACC/Ke) ou recomendação de compra/venda.\n\n"
            f"ESTRUTURA OBRIGATÓRIA DO RELATÓRIO:\n"
            f"Inicie com os três itens de cabeçalho:\n"
            f"- **Overall Fundamentals Status:** **Sufficient Data** | **Partial Data** | **Insufficient Data**\n"
            f"- **Financial Statements Coverage ({ticker}):** **Available** | **Partial** | **Insufficient Data (DATA_UNAVAILABLE)**\n"
            f"- **Confidence:** High | Medium | Low (`Low` se demonstrações ou perfil estiverem retidos/indisponíveis)\n\n"
            f"Em seguida, organize obrigatoriamente nas 6 seções abaixo:\n"
            f"### 1. Visão Geral e Múltiplos (`get_fundamentals`)\n"
            f"### 2. Balanço Patrimonial (`get_balance_sheet`)\n"
            f"### 3. Demonstração de Resultados — DRE (`get_income_statement`)\n"
            f"### 4. Demonstração de Fluxo de Caixa — DFC (`get_cashflow`)\n"
            f"### 5. Transações de Insiders e Limitações de Dados (`get_insider_transactions` e falhas/retenções)\n"
            f"### 6. Tabela Resumo de Indicadores e Demonstrações\n"
            f"Inclua ao final uma tabela Markdown obrigatória com as colunas:\n"
            f"`| Métrica / Linha Contábil | Valor Observado | Período / Competência | Moeda e Unidade | Fonte / Ferramenta |`\n"
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
                    " Report strictly what your tools return; never fill missing financial data from general knowledge."
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

        # Guarda pós-turno: se o modelo encerrou sem novas tool_calls e nenhuma
        # ferramenta financeira central retornou dados válidos, forçamos o relatório
        # determinístico de indisponibilidade sem texto livre potencialmente alucinado.
        if not getattr(result, "tool_calls", None) and any_called:
            if not has_core and (all_core_called or not audit["get_insider_transactions"]["has_data"]):
                report = _build_insufficient_data_report(
                    ticker=ticker,
                    trade_date=current_date,
                    audit=audit,
                )
                result = AIMessage(content=report)
            elif report and band != FundamentalsCoverageBand.SUFFICIENT_DATA:
                report = _ensure_partial_header(ticker, report, band)
                result = AIMessage(content=report)

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
