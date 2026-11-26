"""Offline unit tests for the deterministic Brapi (.SA) connector."""

from __future__ import annotations

import unittest
from unittest import mock

from tradingagents.dataflows import config as config_module, date_window, router
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import brapi
from tradingagents.default_config import build_default_config


class BrapiVendorTests(unittest.TestCase):
    def setUp(self):
        config_module._config = build_default_config()

    def tearDown(self):
        config_module._config = build_default_config()

    def test_non_sa_ticker_raises_without_http(self):
        with (
            mock.patch("tradingagents.dataflows.vendors.brapi.requests.get") as get_mock,
            self.assertRaises(NoMarketDataError),
        ):
            brapi.get_fundamentals("NVDA", "2026-11-24")
        get_mock.assert_not_called()

    def test_historical_date_withholds_profile_without_http(self):
        with (
            mock.patch.object(date_window, "get_current_date", return_value="2026-11-24"),
            mock.patch("tradingagents.dataflows.vendors.brapi.requests.get") as get_mock,
        ):
            out = brapi.get_fundamentals("PETR4.SA", "2025-01-15")
        get_mock.assert_not_called()
        self.assertIn("withheld for this date", out)
        self.assertIn("PETR4.SA", out)

    def test_sa_suffix_stripped_and_all_fields_extracted(self):
        payload = {
            "results": [
                {
                    "symbol": "PETR4",
                    "longName": "Petroleo Brasileiro SA Pfd",
                    "currency": "BRL",
                    "regularMarketPrice": 40.55,
                    "regularMarketChange": -0.75,
                    "regularMarketChangePercent": -1.82,
                    "regularMarketTime": "2026-11-24T20:00:00Z",
                    "fiftyTwoWeekHigh": 45.0,
                    "fiftyTwoWeekLow": 31.0,
                    "marketCap": 569288804804,
                    "priceEarnings": 3.72,
                    "earningsPerShare": 10.89,
                    "defaultKeyStatistics": {
                        "priceToBook": 1.04,
                        "bookValue": 39.16,
                        "dividendYield": 0.0901,
                        "netIncomeToCommon": 140421000000,
                        "mostRecentQuarter": "2026-09-30",
                    },
                    "financialData": {
                        "returnOnEquity": 0.2782,
                        "profitMargins": 0.2385,
                        "ebitda": 289816000000,
                        "totalRevenue": 589940000000,
                        "totalDebt": 362505000000,
                        "totalCash": 62636000000,
                    },
                    "summaryProfile": {
                        "sector": "Energia",
                        "industry": "Petróleo e Gás Integrado",
                    },
                    "dividendsData": {
                        "cashDividends": [
                            {
                                "label": "DIVIDENDO",
                                "rate": 1.2104,
                                "approvedOn": "2026-11-23T03:00:00.000Z",
                                "lastDatePrior": "2026-12-07T03:00:00.000Z",
                                "paymentDate": "2026-12-23T03:00:00.000Z",
                            },
                            {
                                "label": "FUTURE_LEAK",
                                "rate": 9.99,
                                "approvedOn": "2026-12-01T03:00:00.000Z",
                            },
                        ]
                    },
                }
            ]
        }
        fake_resp = mock.Mock(status_code=200, content=b"{}", json=lambda: payload)
        with (
            mock.patch.object(date_window, "get_current_date", return_value="2026-11-24"),
            mock.patch(
                "tradingagents.dataflows.vendors.brapi.requests.get", return_value=fake_resp
            ) as get_mock,
            mock.patch("tradingagents.dataflows.vendors.brapi._throttle"),
        ):
            out = brapi.get_fundamentals("PETR4.SA", "2026-11-24")

        called_url = get_mock.call_args.args[0]
        self.assertTrue(called_url.endswith("/quote/PETR4"))
        self.assertIn("# Company Fundamentals for PETR4.SA (Brapi)", out)
        self.assertIn("P/L (Price to Earnings): 3.72", out)
        self.assertIn("P/VP (Price to Book): 1.04", out)
        self.assertIn("ROE (Return on Equity): 27.82%", out)
        self.assertIn("Dividend Yield: 9.01%", out)
        self.assertIn("LPA (EPS / Lucro por Ação): 10.89 BRL", out)
        self.assertIn("VPA (Book Value per Share): 39.16 BRL", out)
        self.assertIn("Margem Líquida (Net Profit Margin): 23.85%", out)
        self.assertIn("Recent Change (%): -1.82%", out)
        self.assertIn("DIVIDENDO", out)
        self.assertNotIn("FUTURE_LEAK", out)

    def test_null_fields_fallback_to_insufficient_data(self):
        payload = {
            "results": [
                {
                    "symbol": "ITSA4",
                    "longName": "Itausa SA",
                    "currency": "BRL",
                    "regularMarketPrice": 12.93,
                    "regularMarketChangePercent": 0.45,
                    "priceEarnings": 7.65,
                    "earningsPerShare": None,
                }
            ]
        }
        fake_resp = mock.Mock(status_code=200, content=b"{}", json=lambda: payload)
        with (
            mock.patch.object(date_window, "get_current_date", return_value="2026-11-24"),
            mock.patch(
                "tradingagents.dataflows.vendors.brapi.requests.get", return_value=fake_resp
            ),
            mock.patch("tradingagents.dataflows.vendors.brapi._throttle"),
        ):
            out = brapi.get_fundamentals("ITSA4.SA", "2026-11-24")

        self.assertIn("P/L (Price to Earnings): 7.65", out)
        self.assertIn("P/VP (Price to Book): Insufficient Data", out)
        self.assertIn("ROE (Return on Equity): Insufficient Data", out)
        self.assertIn("Dividend Yield: Insufficient Data", out)
        self.assertIn("LPA (EPS / Lucro por Ação): Insufficient Data", out)
        self.assertIn("VPA (Book Value per Share): Insufficient Data", out)
        self.assertIn("Margem Líquida (Net Profit Margin): Insufficient Data", out)

    def test_router_falls_back_to_yfinance_on_brapi_401(self):
        config_module.set_config({"tool_vendors": {"get_fundamentals": "brapi,yfinance"}})
        fake_401 = mock.Mock(
            status_code=401,
            content=b'{"error":true,"message":"MISSING_TOKEN"}',
            json=lambda: {"error": True, "message": "MISSING_TOKEN"},
        )
        with (
            mock.patch.object(date_window, "get_current_date", return_value="2026-11-24"),
            mock.patch("tradingagents.dataflows.vendors.brapi.requests.get", return_value=fake_401),
            mock.patch("tradingagents.dataflows.vendors.brapi._throttle"),
            mock.patch.dict(
                router.VENDOR_METHODS["get_fundamentals"],
                {"yfinance": lambda t, d=None: f"# Company Fundamentals for {t}\nFallback OK"},
            ),
        ):
            out = router.route_to_vendor("get_fundamentals", "ITSA4.SA", "2026-11-24")
        self.assertIn("Fallback OK", out)


if __name__ == "__main__":
    unittest.main()
