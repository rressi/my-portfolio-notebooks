import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        prog=Path(__file__).stem
    )
    parser.add_argument(
        "input_file",
        nargs="+",
        help="Input files"
    )

    args = parser.parse_args()
    for input_file in args.input_file:
        import_file(input_file)


def import_file(
        input_file: str | Path,
):
    input_file = Path(input_file).absolute()

    # Try to automatically detect the input delimiter (.;,| or tab)
    df: pd.DataFrame = (
        pd.read_csv(input_file, sep=None, engine="python")
    )

    # 1) Keep only rows where Transaction is "Acquisto" or "Vendita"
    supported_ops = ["Acquisto", "Vendita"]
    df = df[df["Transazioni"].isin(supported_ops)]

    # 2) Make Quantity negative if Transaction is "Vendita"
    mask: pd.Series = (df["Transazioni"] == "Vendita")
    df.loc[mask, "Quantità"] = -df.loc[mask, "Quantità"]

    # 3) Rename columns
    df = df.rename(columns={
        "Data": "date",
        "Quantità": "quantity",
        "Prezzo unit.": "price",
        "Valuta": "currency",
    })

    # 4) Keep only the requested columns in the given order
    df = df[["date", "quantity", "price", "currency"]]

    # 3) Sort by date ascending (assuming day-first format, e.g. DD/MM/YYYY)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # 6) Save to CSV with comma as field separator
    output_file: Path = (
        input_file.parent 
        / f"{input_file.stem}.out.{input_file.suffix}"
    )
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()

