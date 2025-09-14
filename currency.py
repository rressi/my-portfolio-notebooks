from datetime import timedelta
from IPython.display import display
import pandas as pd
import typing as t
import yfinance as yf

def convert(
        data: pd.DataFrame, 
        to_ccy: str,
):
    if "currency" not in data.columns:
        raise ValueError("Column 'currency' is missing")
    if "price" not in data.columns:
        raise ValueError("Column 'price' is missing")
    if (data["currency"] == to_ccy).all():
        return data # There is nothing to convert.

    date_start: str = (
        data.index[0] - timedelta(days=7)
    ).strftime("%Y-%m-%d")
    date_end: str = (
        data.index[-1] + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    to_tz = data.index.tz
    
    cached_rates: dict[str, pd.Series] = {}
    
    def _get_rates(from_ccy: str, to_ccy: str) -> pd.Series | None:
        symbol: str = f"{from_ccy}{to_ccy}=X".upper()
        if symbol in cached_rates:
            return cached_rates[symbol]
        
        ticker: yf.Ticker = yf.Ticker(symbol)
        tz_name: str = ticker.info.get("exchangeTimezoneName")

        df: pd.DataFrame = (
            ticker.history(
                start=pd.Timestamp(date_start).tz_localize(tz_name), 
                end=pd.Timestamp(date_end).tz_localize(tz_name), 
                interval="1d",
            ).sort_index()
        ).tz_convert(to_tz)

        if df.empty:
            cached_rates[symbol] = None
            return None

        rates: pd.Series = df["Close"].asfreq("D").ffill()
        assert isinstance(rates, pd.Series), f"Invalid type: {rates=}"
        rates.index = rates.index.strftime("%Y-%m-%d")

        cached_rates[symbol] = rates
        return rates
    
    date: pd.Datetime
    row: t.Mapping[str, float | str]
    new_prices: t.Sequence[float] = []
    for date, row in data.iterrows():
        price: float = row["price"]
        if pd.isna(price):
            new_prices.append(pd.NA)
            continue

        from_ccy: str = row["currency"].upper().strip()
        if from_ccy == to_ccy:
            new_prices.append(price)
            continue

        date_s: str = date.strftime("%Y-%m-%d")

        rate: float | None = None
        rates: pd.Series | None = _get_rates(from_ccy, to_ccy)
        if isinstance(rates, pd.Series):
            rate = rates[date_s]
        else:
            rates = _get_rates(to_ccy, from_ccy)
            if isinstance(rate, pd.Series):
                rate = 1 / rates[date_s]
        if rate is None:
            raise ValueError(
                f"Unsupporte currency conversion '{from_ccy}' -> '{to_ccy}'"
            )

        new_prices.append(price * rate)

    data = data.copy()
    data["currency"] = to_ccy
    data["price"] = new_prices
    return data
