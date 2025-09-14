from IPython.display import display
import pandas as pd
import yfinance as yf

import workflow


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

def find_buy_opportunities():

    results = []

    for ticker_name in tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_name)
        company_name: str = workflow.get_company_name(ticker)
        # print(f"Processing {ticker_name} - {company_name}...")

        df: pd.DataFrame = (
            ticker.history(
                period="1mo",
                interval="1d",
            )
        )
        if len(df) < 20:
            print(f"{ticker_name}: not enought data points ({len(df)})")
            continue  # troppo pochi dati

        ma20 = df['Close'].rolling(window=20).mean()
        last_close = df['Close'].iloc[-1]
        last_ma20 = ma20.iloc[-1]
        if pd.isna(last_ma20):
            print(f"{ticker_name}: last_ma20 is NaN")
            continue

        diff_pct = (last_close - last_ma20) / last_ma20 * 100
        results.append({
            'Ticker': ticker_name,
            'Company': company_name,
            'Close': round(last_close, 2),
            'MA20': round(last_ma20, 2),
            '% from MA20': round(diff_pct, 2)
        })

    # crea tabella ordinata per scostamento (dal più negativo)
    table = pd.DataFrame(results).sort_values('% from MA20')
    # display(table)

    candidates = table[table['% from MA20'] <= -1]
    display(candidates)

    for ticker, company, pct_from_ma20 in zip(
        candidates["Ticker"], 
        candidates["Company"],
        candidates["% from MA20"]
    ):
        print(f"{ticker} - {company} - buy score: {-pct_from_ma20}%")
        (
            workflow.WorkflowContext(security=ticker)
            .load()
            .append_last_price()
            .compute_sma()
            .compute_enter_prices()
            .compute_exit_prices()
            .print_last_prices()
            .print_scores()
            .plot()
        )


if __name__ == "__main__":
    find_buy_opportunities()
