"""Deterministic Brapi connector for Brazilian B3 (.SA) fundamentals & quote snapshots."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import UTC, datetime
from typing import Annotated, Any

import requests

from tradingagents.dataflows.date_window import withhold_live_profile
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorUnavailableError,
)

logger = logging.getLogger(__name__)

BRAPI_BASE_URL = os.getenv("BRAPI_BASE_URL", "https://brapi.dev/api").rstrip("/")
_DEFAULT_MIN_INTERVAL_MS = 400
_REQUEST_TIMEOUT_SECONDS = 15

_rate_lock = threading.Lock()
_last_request_ts = 0.0


def _throttle() -> None:
    """Space concurrent Brapi requests across parallel analyst threads."""
    global _last_request_ts
    try:
        min_interval_ms = max(
            0, int(os.getenv("BRAPI_MIN_INTERVAL_MS", str(_DEFAULT_MIN_INTERVAL_MS)))
        )
    except ValueError:
        min_interval_ms = _DEFAULT_MIN_INTERVAL_MS
    min_interval_s = min_interval_ms / 1000.0
    if min_interval_s <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_ts
        if _last_request_ts > 0 and elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        _last_request_ts = time.monotonic()


def to_brapi_symbol(ticker: str) -> tuple[str, str]:
    """Validate and convert a B3 `.SA` ticker into `(canonical, brapi_ticker)`.

    Raises ``NoMarketDataError`` before opening any network socket for
    non-`.SA` symbols so international tickers pass through to other vendors
    with zero latency.
    """
    raw = str(ticker or "").strip()
    canonical = raw.upper()
    if not canonical.endswith(".SA") or len(canonical) <= 3:
        raise NoMarketDataError(ticker, canonical or ticker, "not a B3 .SA listing")
    return canonical, canonical[:-3]


def _get_brapi_token() -> str:
    return (os.getenv("BRAPI_TOKEN") or os.getenv("BRAPI_API_KEY") or "").strip()


def _fetch_brapi_json(
    url: str, params: dict[str, str], canonical: str
) -> tuple[int, dict[str, Any]]:
    _throttle()
    token = _get_brapi_token()
    headers = {"User-Agent": "TradingAgents/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Use requests.get via helper semantics when query token isn't needed;
        # keep secret scrubbed if ever present in errors.
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        msg = str(exc).replace(token, "***") if token else str(exc)
        raise VendorUnavailableError(f"Brapi request failed for {canonical}: {msg}") from None

    try:
        payload = resp.json() if resp.content else {}
    except ValueError:
        payload = {}

    return resp.status_code, payload if isinstance(payload, dict) else {}


def _fmt_num(val: Any, suffix: str = "", decimals: int = 2) -> str:
    if val is None or isinstance(val, bool):
        return "Insufficient Data"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "Insufficient Data"
    formatted = f"{num:.{decimals}f}"
    return f"{formatted}{suffix}"


def _fmt_ratio_and_pct(val: Any) -> str:
    """Format fractional ratios (e.g. 0.2782 -> '27.82% (0.2782)') or mark Insufficient Data."""
    if val is None or isinstance(val, bool):
        return "Insufficient Data"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "Insufficient Data"
    pct = num * 100.0 if abs(num) <= 2.0 else num
    return f"{pct:.2f}% (raw: {num:.4f})"


def _fmt_dividend_yield(stats: dict[str, Any], result: dict[str, Any]) -> str:
    val = stats.get("dividendYield")
    if val is None:
        val = stats.get("yield")
    if val is None:
        val = result.get("dividendYield")
    if val is None or isinstance(val, bool):
        return "Insufficient Data"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "Insufficient Data"
    # Brapi defaultKeyStatistics.dividendYield is fractional (e.g. 0.0901 = 9.01%)
    pct = num * 100.0 if abs(num) <= 1.0 else num
    return f"{pct:.2f}% (raw: {num:.4f})"


def _fmt_money(val: Any, currency: str) -> str:
    if val is None or isinstance(val, bool):
        return "Insufficient Data"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "Insufficient Data"
    cur = f" {currency}" if currency else ""
    if abs(num) >= 1_000_000:
        return f"{num:,.2f}{cur}"
    return f"{num:.2f}{cur}"


def _fmt_dividends(dividends_data: Any, as_of_date: str | None, currency: str) -> list[str]:
    if not isinstance(dividends_data, dict):
        return ["- Proventos recentes (cashDividends): Insufficient Data"]
    cash = dividends_data.get("cashDividends")
    if not isinstance(cash, list) or not cash:
        return ["- Proventos recentes (cashDividends): Insufficient Data"]

    cutoff = str(as_of_date).strip() if as_of_date else None
    filtered: list[dict[str, Any]] = []
    for item in cash:
        if not isinstance(item, dict):
            continue
        approved = str(item.get("approvedOn") or "")[:10]
        if cutoff and approved and approved > cutoff:
            continue
        filtered.append(item)

    if not filtered:
        return ["- Proventos recentes (cashDividends): Insufficient Data"]

    lines = []
    for item in filtered[:5]:
        label = item.get("label") or "PROVENTO"
        rate = _fmt_num(item.get("rate"), f" {currency}" if currency else "", decimals=4)
        approved = str(item.get("approvedOn") or "")[:10] or "Insufficient Data"
        data_com = str(item.get("lastDatePrior") or "")[:10] or "Insufficient Data"
        pay_date = str(item.get("paymentDate") or "")[:10] or "Insufficient Data"
        lines.append(
            f"- {label}: valor/ação={rate} | aprovado em={approved} | data-com={data_com} | pagamento={pay_date}"
        )
    return lines


def get_fundamentals(
    ticker: Annotated[str, "B3 ticker symbol ending in .SA, e.g. ITSA4.SA or PETR4.SA"],
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Fetch deterministic B3 fundamentals & quote summary from Brapi."""
    canonical, b3_symbol = to_brapi_symbol(ticker)

    withheld = withhold_live_profile(curr_date, canonical)
    if withheld:
        return withheld

    url = f"{BRAPI_BASE_URL}/quote/{b3_symbol}"
    full_params = {
        "fundamental": "true",
        "dividends": "true",
        "modules": "defaultKeyStatistics,financialData,summaryProfile",
    }
    base_params = {
        "fundamental": "true",
        "dividends": "true",
    }

    status, payload = _fetch_brapi_json(url, full_params, canonical)
    if status in (400, 401, 402, 403):
        # Retry base fundamental+dividends endpoint in case `modules` is gated by plan.
        base_status, base_payload = _fetch_brapi_json(url, base_params, canonical)
        if base_status == 200:
            status, payload = base_status, base_payload

    if status in (401, 402, 403, 429) or status >= 500:
        msg = payload.get("message") or payload.get("code") or f"HTTP {status}"
        raise VendorUnavailableError(f"Brapi unavailable for {canonical} (HTTP {status}: {msg})")
    if status == 404:
        raise NoMarketDataError(ticker, canonical, "symbol not found in Brapi")
    if status != 200:
        raise VendorUnavailableError(f"Brapi returned unexpected HTTP {status} for {canonical}")

    results = payload.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise NoMarketDataError(ticker, canonical, "empty results from Brapi")

    r = results[0]
    stats = r.get("defaultKeyStatistics") if isinstance(r.get("defaultKeyStatistics"), dict) else {}
    fin = r.get("financialData") if isinstance(r.get("financialData"), dict) else {}
    profile = r.get("summaryProfile") if isinstance(r.get("summaryProfile"), dict) else {}

    currency = str(r.get("currency") or "BRL").strip() or "BRL"
    company_name = r.get("longName") or r.get("shortName") or "Insufficient Data"
    sector = profile.get("sector") or "Insufficient Data"
    industry = profile.get("industry") or "Insufficient Data"

    price = _fmt_money(r.get("regularMarketPrice"), currency)
    change_abs = _fmt_money(r.get("regularMarketChange"), currency)
    change_pct = _fmt_num(r.get("regularMarketChangePercent"), "%", decimals=2)
    market_time = r.get("regularMarketTime") or "Insufficient Data"
    high_52w = _fmt_money(r.get("fiftyTwoWeekHigh"), currency)
    low_52w = _fmt_money(r.get("fiftyTwoWeekLow"), currency)
    market_cap = _fmt_money(
        r.get("marketCap") if r.get("marketCap") is not None else stats.get("marketCap"), currency
    )

    pe_ratio = _fmt_num(
        r.get("priceEarnings") if r.get("priceEarnings") is not None else stats.get("trailingPE")
    )
    pb_ratio = _fmt_num(stats.get("priceToBook"))
    roe = _fmt_ratio_and_pct(fin.get("returnOnEquity"))
    div_yield = _fmt_dividend_yield(stats, r)
    lpa = _fmt_money(
        r.get("earningsPerShare")
        if r.get("earningsPerShare") is not None
        else stats.get("trailingEps"),
        currency,
    )
    vpa = _fmt_money(stats.get("bookValue"), currency)
    net_margin = _fmt_ratio_and_pct(
        fin.get("profitMargins")
        if fin.get("profitMargins") is not None
        else stats.get("profitMargins")
    )

    core_values = [pe_ratio, pb_ratio, roe, div_yield, lpa, vpa, net_margin, change_pct]
    if all(v == "Insufficient Data" for v in core_values) and price == "Insufficient Data":
        raise NoMarketDataError(
            ticker, canonical, "all Brapi fundamental and quote fields are null"
        )

    lines = [
        f"# Company Fundamentals for {canonical} (Brapi)",
        f"Source: Brapi ({url})",
        f"Ticker B3 Consulted: {b3_symbol}",
        f"Snapshot Timestamp (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Company Name: {company_name}",
        f"Sector: {sector}",
        f"Industry: {industry}",
        f"Currency: {currency}",
        "",
        "## Cotação e Variação Recente",
        f"Current Price: {price}",
        f"Recent Change (%): {change_pct}",
        f"Recent Change (Abs): {change_abs}",
        f"Market Quote Time: {market_time}",
        f"52 Week High: {high_52w}",
        f"52 Week Low: {low_52w}",
        "",
        "## Múltiplos Fundamentalistas (B3 / Brapi)",
        f"Market Cap: {market_cap}",
        f"P/L (Price to Earnings): {pe_ratio}",
        f"P/VP (Price to Book): {pb_ratio}",
        f"ROE (Return on Equity): {roe}",
        f"Dividend Yield: {div_yield}",
        f"LPA (EPS / Lucro por Ação): {lpa}",
        f"VPA (Book Value per Share): {vpa}",
        f"Margem Líquida (Net Profit Margin): {net_margin}",
        f"EBITDA: {_fmt_money(fin.get('ebitda'), currency)}",
        f"Receita Total (Total Revenue): {_fmt_money(fin.get('totalRevenue'), currency)}",
        f"Lucro Líquido (Net Income): {_fmt_money(stats.get('netIncomeToCommon'), currency)}",
        f"Dívida Total (Total Debt): {_fmt_money(fin.get('totalDebt'), currency)}",
        f"Caixa Total (Total Cash): {_fmt_money(fin.get('totalCash'), currency)}",
        f"Most Recent Quarter: {stats.get('mostRecentQuarter') or 'Insufficient Data'}",
        "",
        "## Proventos Verificados (Brapi cashDividends)",
        *_fmt_dividends(r.get("dividendsData"), curr_date, currency),
    ]
    return "\n".join(lines)
