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

    csv_paths: t.Sequence[Path] = sorted(data_folder.glob(glob_expr))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching: {data_folder / glob_expr}")

    dfs: t.Sequence[pd.DataFrame] = [
        import_trades(csv_path, sql_path)
        for csv_path in csv_paths
        if csv_path.name != "isin_to_suffix.csv"
    ]

    new_trades: pd.DataFrame = (
        pd.concat(dfs, axis=0)  # keep the timestamp index
        .reset_index()  # bring 'date' back as a column
        # .drop_duplicates() # drop true duplicate rows (incl. date)
        .set_index("date")  # restore DateTimeIndex
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

    df: pd.DataFrame = pd.read_csv(
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
    with sqlite3.connect(":memory:") as con:

        _load_isin_to_suffix_table(
            con=con,
            data_folder=csv_path.parent,
        )

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

    # 1. Convert 'date' to pandas Timestamp
    #    - `utc=True` makes it timezone-aware in UTC
    #    - `dt.tz_localize('local')` will attach your system's local timezone
    trades["date"] = pd.to_datetime(
        trades["date"],
        format="%Y-%m-%d %H:%M:%S",
    )

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


def _load_isin_to_suffix_table(
    con: sqlite3.Connection,
    data_folder: Path | str,
):
    """Load the ISIN to suffix mapping into the SQLite connection."""

    csv_path: Path = Path(data_folder) / "isin_to_suffix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    csv_encoding: str = _detect_encoding(csv_path)

    df: pd.DataFrame = pd.read_csv(
        data_folder / "isin_to_suffix.csv", 
        encoding=csv_encoding,
        engine="python",
        sep=None,
    )

    df.to_sql(
        name="isin_to_suffix",
        con=con,
        if_exists="replace",
        index=False,
    )


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


if __name__ == "__main__":
    data_folder: Path = Path(__file__).parent.parent / "data"
    sql_path: Path = data_folder / "import.sql"

    df = import_many_trades(
        data_folder=data_folder,
        sql_path=sql_path,
        glob_expr="*.csv",
    )
    print(df)
