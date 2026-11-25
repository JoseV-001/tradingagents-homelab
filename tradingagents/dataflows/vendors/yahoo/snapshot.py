"""Verified market snapshot with deterministic technical interpretations."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import price_text
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv


DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema",
    "close_50_sma",
    "close_200_sma",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "macd",
    "macds",
    "macdh",
    "atr",
)


def _verified_rows(symbol: str, as_of_date: str) -> pd.DataFrame:
    data = load_ohlcv(symbol, as_of_date, fill_gaps=False)

    if data is None or data.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), "no price rows")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(as_of_date)].sort_values("Date")

    if df.empty:
        raise NoMarketDataError(
            symbol,
            normalize_symbol(symbol),
            f"no price rows on or before {as_of_date}",
        )

    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return price_text(value)

    return str(value)


def _to_float(value) -> float | None:
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _compare_price(close: float, value: float, label: str) -> tuple[str, str, int]:
    if close > value:
        relation = "ACIMA"
        class_label = f"FECHAMENTO ACIMA DA {label}"
        score = 1
    elif close < value:
        relation = "ABAIXO"
        class_label = f"FECHAMENTO ABAIXO DA {label}"
        score = -1
    else:
        relation = "IGUAL"
        class_label = f"FECHAMENTO IGUAL À {label}"
        score = 0

    return (
        f"FECHAMENTO VS {label}: {relation}. "
        f"Fechamento={_fmt(close)}; {label}={_fmt(value)}.",
        class_label,
        score,
    )


def _build_deterministic_analysis(
    close: float | None,
    raw_values: dict[str, float | None],
) -> list[str]:
    lines = [
        "",
        "## CLASSIFICAÇÕES DETERMINÍSTICAS — CALCULADAS PELO PYTHON",
        "",
        "Estas classificações são calculadas matematicamente. "
        "O LLM não deve reinterpretá-las, modificá-las ou contradizê-las.",
        "",
    ]

    trend_items = (
        ("close_10_ema", "EMA 10", "CURTO PRAZO"),
        ("close_50_sma", "SMA 50", "MÉDIO PRAZO"),
        ("close_200_sma", "SMA 200", "LONGO PRAZO"),
    )

    trend_scores = []
    ma_classifications: dict[str, str] = {}
    lines.append("### TENDÊNCIA — COMPARAÇÕES DE PREÇO")

    if close is None:
        lines.append("- Fechamento indisponível. Comparações não calculadas.")
        for key, _, _ in trend_items:
            ma_classifications[key] = "INDISPONÍVEL"
    else:
        for key, label, period in trend_items:
            value = raw_values.get(key)
            if value is None:
                lines.append(f"- {period}: {label} indisponível.")
                ma_classifications[key] = "INDISPONÍVEL"
                continue

            result, class_label, score = _compare_price(close, value, label)
            ma_classifications[key] = class_label
            lines.append(f"- {period}: {result}")
            lines.append(f"- CLASSIFICAÇÃO {label}: {class_label}")
            if score != 0:
                trend_scores.append(score)

    lines += ["", "### CENÁRIO GERAL — CLASSIFICAÇÃO PYTHON"]

    if not trend_scores:
        scenario = "INDETERMINADO"
    elif all(score > 0 for score in trend_scores):
        scenario = "ALTISTA"
    elif all(score < 0 for score in trend_scores):
        scenario = "BAIXISTA"
    elif any(score > 0 for score in trend_scores) and any(
        score < 0 for score in trend_scores
    ):
        scenario = "MISTO"
    else:
        scenario = "NEUTRO"

    lines.append(f"- CENÁRIO GERAL: {scenario}")

    # RSI
    lines += ["", "### RSI — CLASSIFICAÇÃO PYTHON"]
    rsi = raw_values.get("rsi")

    if rsi is None:
        rsi_class = "INDISPONÍVEL"
    elif rsi < 30:
        rsi_class = "SOBREVENDA"
    elif rsi > 70:
        rsi_class = "SOBRECOMPRA"
    else:
        rsi_class = "NEUTRO"

    lines.append(f"- RSI: {_fmt(rsi)}")
    lines.append(f"- CLASSIFICAÇÃO RSI: {rsi_class}")

    # MACD
    lines += ["", "### MACD — CLASSIFICAÇÃO PYTHON"]

    macd = raw_values.get("macd")
    signal = raw_values.get("macds")
    histogram = raw_values.get("macdh")

    if macd is None:
        macd_class = "INDISPONÍVEL"
    elif macd > 0:
        macd_class = "POSITIVO (ACIMA DE ZERO)"
    elif macd < 0:
        macd_class = "NEGATIVO (ABAIXO DE ZERO)"
    else:
        macd_class = "ZERO"

    lines.append(f"- MACD: {_fmt(macd)}")
    lines.append(f"- MACD VS ZERO: {macd_class}")

    if macd is None or signal is None:
        macd_relation = "INDETERMINADO"
    elif macd > signal:
        macd_relation = "ACIMA DA LINHA DE SINAL"
    elif macd < signal:
        macd_relation = "ABAIXO DA LINHA DE SINAL"
    else:
        macd_relation = "IGUAL À LINHA DE SINAL"

    lines.append(f"- LINHA DE SINAL: {_fmt(signal)}")
    lines.append(f"- RELAÇÃO MACD/SINAL: {macd_relation}")

    if histogram is None:
        hist_class = "INDISPONÍVEL"
    elif histogram > 0:
        hist_class = "POSITIVO"
    elif histogram < 0:
        hist_class = "NEGATIVO"
    else:
        hist_class = "ZERO"

    lines.append(f"- HISTOGRAMA: {_fmt(histogram)}")
    lines.append(f"- CLASSIFICAÇÃO HISTOGRAMA: {hist_class}")

    # Bollinger Bands
    lines += ["", "### BANDAS DE BOLLINGER — CLASSIFICAÇÃO PYTHON"]

    middle = raw_values.get("boll")
    upper = raw_values.get("boll_ub")
    lower = raw_values.get("boll_lb")

    lines.append(f"- BANDA INFERIOR: {_fmt(lower)}")
    lines.append(f"- BANDA MÉDIA: {_fmt(middle)}")
    lines.append(f"- BANDA SUPERIOR: {_fmt(upper)}")
    lines.append(f"- FECHAMENTO: {_fmt(close)}")

    if close is None or middle is None or upper is None or lower is None:
        boll_class = "INDETERMINADO — DADOS INCOMPLETOS"
    elif close > upper:
        boll_class = "ACIMA DA BANDA SUPERIOR"
    elif close < lower:
        boll_class = "ABAIXO DA BANDA INFERIOR"
    elif close > middle:
        boll_class = "ENTRE A BANDA MÉDIA E A SUPERIOR"
    elif close < middle:
        boll_class = "ENTRE A BANDA INFERIOR E A MÉDIA"
    else:
        boll_class = "SOBRE A BANDA MÉDIA"

    lines.append(f"- POSIÇÃO NAS BANDAS: {boll_class}")

    # ATR
    lines += ["", "### ATR — VOLATILIDADE"]

    atr = raw_values.get("atr")
    lines.append(f"- ATR ABSOLUTO: {_fmt(atr)}")

    if atr is not None and close is not None and close != 0:
        atr_pct = (atr / close) * 100
        atr_pct_str = f"{atr_pct:.2f}%"
        atr_vol_class = "NÃO CLASSIFICADA (sem série histórica comparativa)"
        lines.append(f"- ATR/PREÇO: {atr_pct_str}")
        lines.append(
            "- CLASSIFICAÇÃO DE VOLATILIDADE: NÃO CLASSIFICADA. "
            "Não existe comparação histórica calculada neste snapshot."
        )
    else:
        atr_pct_str = "INDISPONÍVEL"
        atr_vol_class = "INDETERMINADA"
        lines.append("- ATR/PREÇO: INDISPONÍVEL")
        lines.append("- CLASSIFICAÇÃO DE VOLATILIDADE: INDETERMINADA")

    macd_combined_class = (
        "INDISPONÍVEL"
        if macd is None
        else f"{macd_class}; {macd_relation}"
    )
    signal_class = (
        "INDISPONÍVEL"
        if signal is None
        else f"LINHA DE REFERÊNCIA ({macd_relation})"
    )

    lines += [
        "",
        "### TABELA DETERMINÍSTICA DE CLASSIFICAÇÕES PYTHON (COPIAR NA TABELA DO RELATÓRIO)",
        "",
        "| Indicador | Valor observado | Classificação Python |",
        "|---|---:|---|",
        f"| Fechamento (Close) | {_fmt(close)} | CENÁRIO GERAL: {scenario} |",
        f"| EMA 10 (close_10_ema) | {_fmt(raw_values.get('close_10_ema'))} | {ma_classifications['close_10_ema']} |",
        f"| SMA 50 (close_50_sma) | {_fmt(raw_values.get('close_50_sma'))} | {ma_classifications['close_50_sma']} |",
        f"| SMA 200 (close_200_sma) | {_fmt(raw_values.get('close_200_sma'))} | {ma_classifications['close_200_sma']} |",
        f"| RSI (rsi) | {_fmt(rsi)} | {rsi_class} |",
        f"| MACD (macd) | {_fmt(macd)} | {macd_combined_class} |",
        f"| Linha de Sinal MACD (macds) | {_fmt(signal)} | {signal_class} |",
        f"| Histograma MACD (macdh) | {_fmt(histogram)} | {hist_class} |",
        f"| Banda Superior Bollinger (boll_ub) | {_fmt(upper)} | LIMITE SUPERIOR ({boll_class}) |",
        f"| Banda Média Bollinger (boll) | {_fmt(middle)} | BANDA CENTRAL ({boll_class}) |",
        f"| Banda Inferior Bollinger (boll_lb) | {_fmt(lower)} | LIMITE INFERIOR ({boll_class}) |",
        f"| ATR Absoluto (atr) | {_fmt(atr)} | {atr_vol_class} |",
        f"| ATR / Preço | {atr_pct_str} | {atr_vol_class} |",
        "",
        "### REGRAS PARA O LLM",
        "- Use estas classificações exatamente como fornecidas; copie a coluna `Classificação Python` da tabela acima sem substituí-la por `N/A`.",
        "- Não altere nem recalcule as classificações.",
        "- Não chame EMA 10, SMA 50, SMA 200 ou Bandas de Bollinger de 'suporte confirmado', 'suporte dinâmico', 'resistência confirmada' ou 'piso/teto validado': limite-se estritamente ao fato calculado (fechamento ACIMA, ABAIXO ou IGUAL à média/banda).",
        "- Não invente níveis de suporte ou resistência.",
        "- Não declare reversões sem evidência histórica.",
        "- Não classifique volatilidade como baixa, média ou alta sem comparação.",
        "- Cenário geral é uma classificação das relações das médias, não uma previsão de preço.",
    ]

    return lines


def build_verified_market_snapshot(
    symbol: str,
    as_of_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render verified OHLCV, indicators, and deterministic classifications."""

    df = _verified_rows(symbol, as_of_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)

    indicator_values: dict[str, str] = {}
    raw_values: dict[str, float | None] = {}

    for name in selected:
        try:
            stock_df[name]
            raw = stock_df.iloc[-1][name]
            indicator_values[name] = _fmt(raw)
            raw_values[name] = _to_float(raw)
        except Exception as exc:
            indicator_values[name] = f"N/A ({type(exc).__name__})"
            raw_values[name] = None

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    close = _to_float(latest.get("Close"))

    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {as_of_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]

    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += [
        "",
        "### Verified technical indicators (latest row)",
        "",
        "| Indicator | Value |",
        "|---|---:|",
    ]

    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines.extend(_build_deterministic_analysis(close, raw_values))

    lines += [
        "",
        f"### Recent verified closes (last {len(recent)} rows)",
        "",
        "| Date | Close |",
        "|---|---:|",
    ]

    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]

    return "\n".join(lines)
