from colorama import Fore, Style
from IPython.display import display
import matplotlib.pyplot as plt
import math
import pandas as pd
import typing as t
import yfinance as yf


class WorkflowContext(t.NamedTuple):
    security: str

    buy_column: str | None = None
    buy_date: str | pd.Timestamp | None = None
    buy_price: float | None = None
    company_name: str | None = None
    data: pd.DataFrame | None = None
    entry_column: str | None = None
    entry_prices: t.Sequence[float] | None = None
    last_date: pd.Timestamp | None = None
    last_price: float | None = None
    last_sma: float | None = None
    sma_column: str | None = None
    sma_lenght: int = 20
    start_date: str | pd.Timestamp = "2025-06-01"
    target_column: str | None = None
    target_prices: t.Sequence[float] | None = None
    time_zone: str = "America/New_York"

    def load(self) -> t.Self:
        ticker: yf.Ticker = yf.Ticker(self.security)
        company_name: str = get_company_name(ticker)

        start_date: pd.Timestamp = (
            pd.Timestamp(self.start_date)
            .tz_localize(self.time_zone)
        )
        data: pd.DataFrame = (
            ticker.history(
                start=self.start_date, 
                interval="1d",
            )
        )

        last_date: pd.Timestamp = data.index[-1]
        last_price: float | None = first_value(
            ticker.fast_info["last_price"],
            data["Close"].iloc[-1],
        )
        assert isinstance(last_price, float | None)

        return self._replace(
            company_name=company_name,
            data=data,
            last_date=last_date,
            last_price=last_price,
            start_date=start_date,
        )

    def compute(self) -> t.Self:
        data: pd.DataFrame = self.data
        assert isinstance(data, pd.DataFrame)

        sma_column: str =f"SMA-{self.sma_lenght}"
        data[sma_column] = data["Close"].rolling(
            window=self.sma_lenght,
        ).mean()

        last_sma: float = first_value(
            data[sma_column].iloc[-1],
        )
        assert isinstance(last_sma, float | None)

        buy_column: str = "Buy Price"
        buy_date: pd.Timestamp | None = None
        if self.buy_price is not None:
            data[buy_column] = self.buy_price
            buy_date: pd.Timestamp = (
                pd.Timestamp(self.buy_date)
                .tz_localize(self.time_zone)
            )
            if buy_date not in data.index:
                idx = data.index.get_indexer([buy_date], method="pad")[0]
                if idx >= 0:
                    buy_date = data.index[idx]

        entry_column: str = "Buy"        
        entry_prices: t.Sequence[float] | None = (
            self.entry_prices if self.entry_prices is not None 
            else derive_prices(
                reference_price=first_value(
                    last_sma,
                    self.last_price,
                ),
                index_first=-1,
                index_last=-4,
                ratio=1.05,
            )
        )
        if entry_prices:
            for i, entry_price in enumerate(entry_prices):
                data[f"{entry_column} #{i + 1}"] = entry_price

        target_column: str = "Sell"        
        target_prices: t.Sequence[float] | None = (
            self.target_prices if self.target_prices is not None 
            else derive_prices(
                reference_price=first_value(
                    self.buy_price,
                    last_sma,
                    self.last_price,
                ),
                index_first=1,
                index_last=4,
                ratio=1.05,
            )
        )
        if target_prices:
            for i, target_price in enumerate(target_prices):
                data[f"{target_column} #{i + 1}"] = target_price

        return self._replace(
            buy_date=buy_date,
            buy_column=buy_column,
            data=data,
            entry_prices=entry_prices,
            entry_column=entry_column,
            last_sma=last_sma,
            sma_column=sma_column,
            target_column=target_column,
            target_prices=target_prices,
        )

    def print_prices(self) -> t.Self:

        def _represent_price(
            pos: int,
            price: float | None,
        ) -> str:
            if price is None:
                return "N/A"
            if isinstance(price, float):
                if pos < 0:
                    return f"{Fore.GREEN}{price:,.2f}{Style.RESET_ALL}"
                if pos == 0:
                    return f"{price:.4f}"
                if pos > 0:
                    return f"{Fore.YELLOW}{price:,.2f}{Style.RESET_ALL}"
            return str(price)

        prices: t.Sequence[str] = [
            *(
                _represent_price(-1, entry_price) 
                for entry_price in sorted(
                    self.entry_prices or []
                )
            ),
            _represent_price(
                pos=0,
                price=first_value(
                    self.buy_price,
                    self.last_sma,
                    self.last_price,
                ),
            ),
            *(
                _represent_price(1, target_price) 
                for target_price in sorted(
                    self.target_prices or []
                )
            ),
        ]
        print("Prices:", " | ".join(prices))

        return self

    def print_scores(self) -> t.Self:

        data: pd.DataFrame = self.data
        assert isinstance(data, pd.DataFrame)

        last_sma: float = self.last_sma
        last_price: float = self.last_price

        buy_score: float = 100 * (
            (last_sma - last_price)
            / last_price
        )
        buy_color: str = Fore.GREEN if buy_score >= 5 else Fore.LIGHTWHITE_EX
        print(f"{buy_color}Buy score: {buy_score:.2f}%")

        if self.buy_price is not None:
            buy_price: float = self.buy_price        
            sell_score: float = 100 * (last_price - buy_price) / buy_price
            sell_color = Fore.YELLOW if sell_score >= 5 else Fore.LIGHTWHITE_EX
            print(f"{sell_color}Sell score: {sell_score:.2f}%")

        return self

    def plot(self) -> t.Self:
        data: pd.DataFrame = self.data
        assert isinstance(data, pd.DataFrame)

        plt.figure(figsize=(12,6))

        # Market prices:
        if self.last_price is not None:
            close_price: float = data["Close"].iloc[-1]
            plt.plot(
                data.index, 
                data["Close"], 
                label=f"Close price: {close_price:.2f}",
                linewidth=1, color="blue",
            )
            plt.scatter(
                [self.last_date],
                [self.last_price],
                s=80, zorder=3, color="blue",
            )
            plt.annotate(
                f"{self.last_price:.2f}", 
                (self.last_date, self.last_price), 
                xytext=(10,-15), textcoords="offset points", color="blue",
            )

        # SMA-X:
        if self.last_sma is not None:
            plt.plot(
                data.index, 
                data[self.sma_column], 
                label=f"{self.sma_column}: {self.last_sma:.2f}",
                linewidth=2, color="orange",
            )
            plt.scatter(
                [self.last_date],
                [self.last_sma],
                s=80, zorder=3, color="orange",
            )
            plt.annotate(
                f"{self.last_sma:.2f}", 
                (self.last_date, self.last_sma), 
                xytext=(10,-15), textcoords="offset points", color="orange",
            )

        # Buy price:
        if self.buy_price is not None:
            plt.plot(
                data.index, 
                data[self.buy_column], 
                label=f"{self.buy_column}: {self.buy_price:.2f}", 
                linewidth=2,
                linestyle="--", color="red",
            )
            plt.scatter(
                [self.buy_date],
                [self.buy_price],
                s=80, zorder=3, color="red",
            )
            plt.annotate(
                f"{self.buy_price:.2f}", 
                (self.buy_date, self.buy_price), 
                xytext=(10,10), textcoords="offset points", color="red",
            )

        # Entry prices:
        if self.entry_prices:
            for i, price in enumerate(self.entry_prices):
                self.plot_price_level(
                    color="forestgreen",
                    column=f"{self.entry_column} #{i + 1}",
                    price=price,
                )

        # Target prices:
        if self.target_prices:
            for i, price in enumerate(self.target_prices):
                self.plot_price_level(
                    color="gold",
                    column=f"{self.target_column} #{i + 1}",
                    price=price,
                )

        plt.title(
            f"{self.security} ({self.company_name}) - "
            f"Close price with SMA({self.sma_lenght})"
        )
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend(
            loc="center left", 
            bbox_to_anchor=(1e-2, 0.5),
        )
        plt.grid(True)
        plt.show()

        return self

    def plot_price_level(
        self,
        column: str,
        price: float,
        color: str,
    ):
        plt.plot(
            self.data.index,
            self.data[column],
            label=f"{column}: {price:.2f}",
            linewidth=1,
            color=color,
            linestyle="--",
        )

    def show_company_table(self) -> t.Self:
        ticker = yf.Ticker(self.security)
        info = dict(ticker.info)

        revenue: int | None = info.get("totalRevenue", None)
        if revenue is not None:
            info["totalRevenue"] = "{:,d}".format(revenue)

        gross_profits: int | None = info.get("grossProfits", None)
        if gross_profits is not None:
            info["grossProfits"] = "{:,d}".format(gross_profits)

        data = {
            "Company Name": [self.company_name],
            "Industry": [info.get("industry")],
            "Revenue (ttm)": [info.get("totalRevenue")],
            "Gross Profit (ttm)": [info.get("grossProfits")],
            "Current Price": [info.get("currentPrice")],
            "Target Mean Price": [info.get("targetMeanPrice")],
            "Target High Price": [info.get("targetHighPrice")],
            "Target Low Price": [info.get("targetLowPrice")],
            "Trailing EPS": [info.get("trailingEps")],
            "Forward EPS": [info.get("forwardEps")],
            "Website": [info.get("website")],
        }
        df = pd.DataFrame(
            data=data,
            index=[self.security],
        )
        display(df.T)
        return self

    def show_status(self) -> t.Self:
        ticker: yf.Ticker = yf.Ticker(self.security)
        fast_info: dict[str, t.Any] = dict(ticker.fast_info)

        def _get_field(key: str) -> str:
            return key.capitalize().replace("_", " ")

        def _get_value(key: str) -> str | None:
            value: str | float | int | None = fast_info.get(key, None)
            if isinstance(value, float):
                return f"{value:,.2f}"
            if isinstance(value, int):
                return f"{value:,d}"
            return value

        data: dict[str, t.Sequence[str | None]] = {
            _get_field(key): [_get_value(key)]
            for key in fast_info.keys()
        }
        df = pd.DataFrame(
            data=data,
            index=[self.security],
        )
        display(df.T)
        return self


def run(**kwargs):
    (
        WorkflowContext(**kwargs)
        .load()
        .compute()
        .print_prices()
        .print_scores()
        .plot()
        .show_company_table()
        .show_status()
    )


def first_value(
        *seq: float | None,
) -> float | None:
    """Return the first non-None value in a sequence, or None if all are None."""
    return next(
        (float(x) for x in seq if x is not None), 
        None  # Default value if all are None
    )


def get_company_name(
        ticker: yf.Ticker,
) -> str | None:
    company_name: str = ticker.info.get("shortName")
    if company_name is None:
        return None
    if "," in company_name:
        company_name = company_name.split(",")[0].strip()
    if "-" in company_name:
        company_name = company_name.split(" - ")[0].strip()
    company_name = (
        company_name.replace(" Inc.", "")
        .replace(" Corporati", "")
        .replace(" Corporation", "")
        .replace(" Corp.", "")
        .replace(" Incorporated", "")
        .replace(" plc", "")
    )
    return company_name


def derive_prices(
        reference_price: float | None,
        index_first: int = 1,
        index_last: int = 4,
        ratio: float = 1.05,
) -> t.Sequence[float] | None:
    if reference_price is None:
        return None
    range_step: int = 1 if index_first < index_last else -1 
    return [
        reference_price * math.pow(ratio, x)
        for x in range(
            index_first, 
            index_last, 
            range_step,
        )
    ]
