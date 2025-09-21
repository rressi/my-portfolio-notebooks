from datetime import timedelta
from IPython.display import display
import pandas as pd
import typing as t
import yfinance as yf


def to_currency(
    data: pd.DataFrame,
    to_ccy: str,
    selected_columns: t.Sequence[str] = ("price", "cost", "net amount"),
    currency_column: str = "currency",
):
    if currency_column not in data.columns:
        raise ValueError(f"Currency column '{currency_column}' is missing")
    
    if (data[currency_column] == to_ccy).all():
        return data  # There is nothing to convert.

    selected_columns = sorted(
        set(selected_columns) & set(data.columns)
    )
    if not selected_columns:
        return data  # There is nothing to convert.

    date_start: str = (data.index[0] - timedelta(days=7)).strftime("%Y-%m-%d")
    date_end: str = (data.index[-1] + timedelta(days=7)).strftime("%Y-%m-%d")
    to_tz = data.index.tz

    cached_rates: dict[str, pd.Series] = {}

    def _get_rate_history(from_ccy: str, to_ccy: str) -> pd.Series | None:
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

    cached_rates_at: dict[tuple[str, str, str], float] = {}

    def _get_rate_at(from_ccy: str, to_ccy: str, at_date: str) -> float:
        input = (from_ccy, to_ccy, at_date)

        cached_rate: float | None
        if cached_rate := cached_rates_at.get(input):
            return cached_rate

        direct_rates: pd.Series | None = _get_rate_history(from_ccy, to_ccy)
        if isinstance(direct_rates, pd.Series):
            rate: float = direct_rates[at_date]
            cached_rates_at[input] = rate
            return rate

        inverted_rates: pd.Series | None = _get_rate_history(to_ccy, from_ccy)
        if isinstance(direct_rates, pd.Series):
            rate: float = 1 / inverted_rates[at_date]
            cached_rates_at[input] = rate
            return rate

        raise ValueError(
            f"Unsupporte currency conversion '{from_ccy}' -> '{to_ccy}'"
        )

    converted_values: dict[str, list[float]] = {
        column: [] for column in selected_columns
    }

    # Perform the currency conversion for the entire data-set:
    timestamp: pd.Datetime
    row: t.Mapping[str, float | str]
    for timestamp, row in data.iterrows():
        date: str = timestamp.strftime("%Y-%m-%d")
        for column in selected_columns:
            value: float = row[column]
            if pd.isna(value):
                converted_values[column].append(pd.NA)
                continue
            from_ccy: str = row[currency_column].upper().strip()
            if from_ccy == to_ccy:
                converted_values[column].append(value)
                continue
            rate: float = _get_rate_at(
                from_ccy=from_ccy,
                to_ccy=to_ccy,
                at_date=date,
            )
            converted_values[column].append(value * rate)

    # Apply the changes to the data-frame:
    data = data.copy()
    data[currency_column] = to_ccy
    for column in selected_columns:
        data[column] = converted_values[column]

    return data
