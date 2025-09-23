from IPython.display import display
import pandas as pd
import yfinance as yf

from my_portfolio._workflow import (
    Context as WorkflowContext,
    get_company_name,
)


tickers = [
    "AAPL",
    "ADBE",
    "AMD",
    "AMZN",
    "ANET",
    "ARM",
    "ASML",
    "AVGO",
    "AVTR",
    "BTC-EUR",
    "COHR",
    "COIN",
    "CRM",
    "CRWD",
    "CRWV",
    "CSCO",
    "ETH-EUR",
    "GOOGL",
    "IBM",
    "INTC",
    "KLAC",
    "LRCX",
    "META",
    "MPWR",
    "MSFT",
    "MU",
    "NBIS",
    "NOW",
    "NVDA",
    "NVTS",
    "ORCL",
    "PLTR",
    "QCOM",
    "RACE",
    "SMCI",
    "STX",
    "TSLA",
    "TSM",
    "TXN",
    "WDC",
    # "XAU-USD",
    "XRP-EUR",
    "ZGLD.SW",
]


def find_buy_opportunities(
    sma_lenght=5,
):

    results = []

    for ticker_name in tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_name)
        company_name: str = get_company_name(ticker)
        print(f"Processing {ticker_name} - {company_name}...")

        df: pd.DataFrame = ticker.history(
            period="1mo",
            interval="1d",
        )
        if len(df) < sma_lenght:
            print(f"{ticker_name}: not enought data points ({len(df)})")
            continue  # troppo pochi dati

        close: pd.Series = df["Close"]
        sma: pd.Series = close.rolling(
            window=sma_lenght,
        ).mean()

        last_close: float = close.iloc[-1]
        last_sma: float = sma.iloc[-1]
        if pd.isna(last_sma):
            print(f"{ticker_name}: last_sma is NaN")
            continue
        diff_pct: float = (last_close - last_sma) / last_sma * 100

        results.append(
            {
                "Ticker": ticker_name,
                "Company": company_name,
                "Close": round(last_close, 2),
                "SMA": round(last_sma, 2),
                "% from SMA": round(diff_pct, 2),
            }
        )

    # crea tabella ordinata per scostamento (dal più negativo)
    table = pd.DataFrame(results).sort_values("% from SMA")
    # display(table)

    candidates = table[table["% from SMA"] <= -1]
    display(candidates)

    for ticker, company, pct_from_sma in zip(
        candidates["Ticker"], candidates["Company"], candidates["% from SMA"]
    ):
        print(f"{ticker} - {company} - buy score: {-pct_from_sma}%")
        (
            WorkflowContext(
                security=ticker,
                sma_fast_lenght=sma_lenght,
            )
            .load()
            .handle_last_price()
            .compute_SMAs()
            .compute_enter_prices()
            .compute_exit_prices()
            .print_last_prices()
            .print_scores()
            .plot()
        )


if __name__ == "__main__":
    find_buy_opportunities()
