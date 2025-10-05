import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from my_portfolio import (
    import_many_trades,
    to_currency
)


import pandas as pd

def df_to_yahoo_csv(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """
    Convert a DataFrame with datetime index (UTC or tz-aware) and columns:
        ISIN, ticker, price, costs, currency, quantity
    into Yahoo Finance import format:
        Symbol,Trade Date,Purchase Price,Quantity,Comment,Extra

    - Quantity > 0 = buy (costs added to price)
    - Quantity < 0 = sell (costs subtracted from price)
    - Index converted to UTC date (YYYYMMDD)
    - Comment keeps ISIN and currency
    - Extra left blank
    """
    required = {"ISIN", "ticker", "price", "costs", "currency", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Ensure index is datetime and in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    dt = df.index
    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")

    trade_date = dt.strftime("%Y%m%d")

    # Calculate per-share transaction cost
    cost_per_share = df["costs"] / df["quantity"].abs()

    # Adjust the price to include transaction costs
    adj_price = df["price"].copy()
    adj_price[df["quantity"] > 0] = df["price"][df["quantity"] > 0] + cost_per_share[df["quantity"] > 0]
    adj_price[df["quantity"] < 0] = df["price"][df["quantity"] < 0] - cost_per_share[df["quantity"] < 0]

    # Compose Yahoo-compatible DataFrame
    out = pd.DataFrame({
        "Symbol": df["ticker"].astype(str),
        "Trade Date": trade_date,
        "Purchase Price": adj_price.round(6),  # rounded for nicer CSV output
        "Quantity": df["quantity"].astype(float),
        "Comment": (
            "ISIN=" + df["ISIN"].astype(str)
            + ", CCY=" + df["currency"].astype(str)
            + ", OrigPrice=" + df["price"].astype(str)
            + ", Costs=" + df["costs"].astype(str)
        ),
        "Extra": "",
    })

    if out_path:
        out.to_csv(out_path, index=False, encoding="utf-8")

    return out



def filter_open_positions(
    df: pd.DataFrame,
    symbol_col: str = "ticker",
    qty_col: str = "quantity",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Keep ONLY symbols whose FINAL net position is non-zero.
    """
    if symbol_col not in df.columns or qty_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{symbol_col}' and '{qty_col}'")

    final_pos = df.groupby(symbol_col, dropna=False)[qty_col].sum()
    open_syms = final_pos[final_pos.abs() > tolerance].index
    return df[df[symbol_col].isin(open_syms)].copy()


def keep_last_open_segment(
    df: pd.DataFrame,
    symbol_col: str = "ticker",
    qty_col: str = "quantity",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    For each symbol, keep ONLY the transactions AFTER the last point where the
    cumulative quantity was ~0 (i.e., keep the current open leg).
    Symbols ending at ~0 are removed entirely.

    - df.index: DatetimeIndex (naive or tz-aware); function normalizes to UTC.
    - Buys: +quantity, sells: -quantity.
    - Uses per-group slicing + concat (no index-only selection) to avoid
      duplicates when different symbols share the same timestamp.
    """
    if symbol_col not in df.columns or qty_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{symbol_col}' and '{qty_col}'")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    # Normalize index to UTC (avoids tz heterogeneity)
    if df.index.tz is None:
        idx_utc = df.index.tz_localize("UTC")
    else:
        idx_utc = df.index.tz_convert("UTC")

    w = df.copy()
    w.index = idx_utc
    w = w.sort_index(kind="mergesort")

    # Running position per symbol
    w["_cum_qty"] = w.groupby(symbol_col, dropna=False)[qty_col].cumsum()

    # Keep only symbols that end non-zero
    ending_pos = w.groupby(symbol_col, dropna=False)["_cum_qty"].last().fillna(0.0)
    open_syms = ending_pos[ending_pos.abs() > tolerance].index

    keep_frames = []
    for _, g in w[w[symbol_col].isin(open_syms)].groupby(symbol_col, sort=False):
        at_zero = np.isclose(g["_cum_qty"].to_numpy(), 0.0, atol=tolerance)
        zero_pos = np.flatnonzero(at_zero)
        cut = zero_pos[-1] if zero_pos.size else -1  # keep strictly AFTER last zero
        keep_frames.append(g.iloc[cut + 1 :].drop(columns=["_cum_qty"]))

    if not keep_frames:
        return w.iloc[0:0].drop(columns=["_cum_qty"])

    out = pd.concat(keep_frames, axis=0)
    return out.sort_index(kind="mergesort")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export transactions to a CSV file.",
    )
    parser.add_argument(
        "export_path",
        type=str,
        help="Path to the output CSV file.",
        default="exported-transactions.csv",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Export only transactions for currently open positions.",
        default=False,
    )
    parser.add_argument(
        "--last-open-segment",
        action="store_true",
        help="For each symbol, keep only transactions after the last time the position was zero.",
        default=False,
    )
    parser.add_argument(
        "--convert-currency",
        help="Convert all transactions to the given target currency (e.g., USD).",
        default=False,
    )

    args: argparse.Namespace = parser.parse_args()

    trades: pd.DataFrame = import_many_trades(
        data_folder=Path("data"), 
        sql_path=Path("data/import.sql"),
    )

    if args.open_only:
        trades = filter_open_positions(trades)
    if args.last_open_segment:
        trades = keep_last_open_segment(trades)

    if args.convert_currency:
        trades = to_currency(
            trades, 
            target_currency=args.convert_currency,
        )
    df_to_yahoo_csv(trades, out_path=args.export_path)
