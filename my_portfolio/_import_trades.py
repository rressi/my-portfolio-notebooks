import functools
import sqlite3
from pathlib import Path
import typing as t

import chardet
import pandas as pd
import tzlocal


@functools.cache
def import_many_trades(
    data_folder: Path | str,
    sql_path: Path | str,
    glob_expr: str = "*.csv",
) -> pd.DataFrame:
    """
    Import trades from all CSV files matching the glob expression,
    apply the SQL query, and return a single DataFrame with duplicates removed.
    """
    data_folder = Path(data_folder)
    sql_path = Path(sql_path)

    csv_paths: t.Sequence[Path] = list(data_folder.glob(glob_expr))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching: {data_folder / glob_expr}")

    dfs: t.Sequence[pd.DataFrame] = [
        import_trades(csv_path, sql_path) for csv_path in csv_paths
    ]

    new_trades: pd.DataFrame = (
        pd.concat(dfs, axis=0) # keep the timestamp index
            .reset_index()     # bring 'date' back as a column
            # .drop_duplicates() # drop true duplicate rows (incl. date)
            .set_index("date") # restore DateTimeIndex
            .sort_index()
    )
    return new_trades


@functools.cache
def import_trades(
    csv_path: Path | str,
    sql_path: Path | str,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    sql_path = Path(sql_path)

    csv_encoding: str = _detect_encoding(csv_path)

    df = pd.read_csv(
        csv_path,
        encoding=csv_encoding,
        engine="python",
        sep=None,
    )

    # The columns for this DataFrame are:
    #   - date       : transaction timestamp in format 'YYYY-MM-DD hh:mm:ss'
    #   - ticker     : stock symbol
    #   - price      : unit price of the transaction
    #   - costs      : transaction costs/fees
    #   - currency   : transaction currency (e.g., USD)
    #   - quantity   : positive for buys, negative for sells
    trades: pd.DataFrame

    con: sqlite3.Connection = sqlite3.connect(":memory:")
    try:
        df.to_sql(
            name="trades",
            con=con,
            if_exists="replace",
            index=False,
        )
        query: str = sql_path.read_text(
            encoding="utf-8",
        )
        trades = pd.read_sql_query(query, con)
    finally:
        con.close()

    # 1. Convert 'date' to pandas Timestamp
    #    - `utc=True` makes it timezone-aware in UTC
    #    - `dt.tz_localize('local')` will attach your system's local timezone
    trades["date"] = pd.to_datetime(trades["date"], format="%Y-%m-%d %H:%M:%S")

    # If the column is naive, localize it to your system's timezone:
    local_timezone: str = tzlocal.get_localzone_name()
    trades["date"] = trades["date"].dt.tz_localize(
        tz=local_timezone,
    )

    # 2. Set 'date' as index
    trades = trades.set_index("date")

    # 3. Sort by this new index
    trades = trades.sort_index()

    return trades


def _detect_encoding(
    file_path: Path | str,
    sample_size=100000,
) -> str:
    """Detect file encoding by reading a sample of bytes."""

    raw_data: bytes
    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)

    result = chardet.detect(raw_data)
    return result["encoding"]
