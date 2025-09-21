from colorama import Fore, Style
import enum
import io
from pathlib import Path
import typing as t

from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from my_portfolio._import_trades import (
    import_many_trades,
)
from my_portfolio._currency import (
    to_currency as convert_currency,
)


class Column(enum.StrEnum):
    DATE="date"
    ENTER = "enter"
    EXIT = "exit"
    PRICE = "price"  # Market price
    PURCHASE = "purchase"  # Avg. purchase price
    QUANTITY = "quantity"
    SMA_FAST = "sma"
    SMA_SLOW = "sma-slow"


class Context(t.NamedTuple):
    security: str
    
    company_name: str = ""
    data: pd.DataFrame = pd.DataFrame()
    invest_ratio: float = 1.05
    last_enter_prices: t.Sequence[float] = tuple()
    last_exit_prices: t.Sequence[float] = tuple()
    operations: pd.DataFrame = pd.DataFrame()
    sma_fast_lenght: int = 10
    sma_slow_lenght: int = 20
    start_date: str | pd.Timestamp = "2025-06-01"
    ticker: yf.Ticker | None = None

    def load(self) -> t.Self:
        ticker: yf.Ticker = yf.Ticker(self.security)
        company_name: str = get_company_name(ticker)
        tz_name = ticker.info.get("exchangeTimezoneName")

        start_date: pd.Timestamp = (
            pd.Timestamp(self.start_date)
            .tz_localize(tz_name)
        )
        start_date_before: pd.Timestamp = min(
            start_date - pd.Timedelta(days=self.sma_fast_lenght + 1),
            start_date - pd.Timedelta(days=self.sma_slow_lenght + 1)
        )
        data: pd.DataFrame = (
            ticker.history(
                start=start_date_before, 
                interval="1d",
            )
            [["Close"]]
            .rename(columns={"Close": Column.PRICE.value})
            .sort_index()
        )
        if data.empty:
            raise ValueError(f"No data found for {self.security}")
        
        return self._replace(
            company_name=company_name,
            data=data,
            start_date=start_date,
            ticker=ticker,
        )
    
    def append_last_price(self) -> t.Self:
        if self.data.empty:
            return self

        last_price: float | None = (
            self.ticker.fast_info.get("last_price", None)
        )
        if last_price is None:
            return self

        tz=data.index.tz
        current_date: pd.Timestamp = (
            pd.Timestamp.now(tz=tz)
            .normalize()
        )
        data = pd.concat([
            data,
            pd.DataFrame(
                data={"Price": last_price},
                index=[current_date],
            ),
        ])
        data = data[~data.index.duplicated(keep='last')]

        return self._replace(
            data=data,
        )
    
    def compute_SMAs(self) -> t.Self:
        if self.data.empty:
            return self

        price: pd.Series = self.data[Column.PRICE.value]
        sna_fast: pd.Series = price.rolling(
            window=self.sma_fast_lenght,
        ).mean()
        sna_slow: pd.Series = price.rolling(
            window=self.sma_slow_lenght,
        ).mean()

        data: pd.DataFrame = self.data.copy()
        data[Column.SMA_FAST.value] = sna_fast
        data[Column.SMA_SLOW.value] = sna_slow

        # Remove items before the date of the first valid SMA
        first_valid_sma_idx = max(
            sna_fast.first_valid_index() or -1,
            sna_slow.first_valid_index() or -1
        )
        if first_valid_sma_idx is not None:
            data = data.loc[first_valid_sma_idx:]

        return self._replace(
            data=data,
        )
    
    def import_trades(self) -> t.Self:
        if self.data.empty:
            return self
        
        operations: pd.DataFrame = import_many_trades(
            data_folder=Path("data"),
            sql_path=Path("data/import.sql")
        )

        selected_ticker: str = self.security
        match (
            self.ticker.info.get('quoteType'),
            self.ticker.info.get("fromCurrency")
        ):
            case ['CRYPTOCURRENCY', crypto_ticker]:
                selected_ticker = crypto_ticker

        operations = operations[
            operations["ticker"] == selected_ticker
        ]
        if operations.empty:
            return self
        
        # Converts the timestamps
        operations = operations.tz_convert(
            self.data.index.tz
        )

        if "currency" in operations.columns:
            ticker_cur: str = self.ticker.info["currency"]
            operations = convert_currency(operations, ticker_cur)
        
        data: pd.DataFrame = self.data.copy()
        from_date: pd.Timestamp = (
            # to midnight
            data.index[0].normalize()
        )
        to_date: pd.Timestamp = (
            # to midnight
            data.index[-1].normalize()
            # of the next day
            + pd.Timedelta(days=1)
        )

        col_quantity: str = Column.QUANTITY.value
        col_purchase: str = Column.PURCHASE.value

        # Compute average purchase price:
        avg_price: float = 0.0
        tot_cost: float = 0.0
        tot_quantity: float = 0.0
        date: pd.Timestamp
        row: t.Mapping[str, t.Any]
        for date, row in operations.iterrows():
            price: float = row["price"]
            quantity: float = row["quantity"]
            tot_quantity += quantity
            if tot_quantity > 0.0:
                tot_cost += quantity * price
                avg_price = tot_cost / tot_quantity
            else:
                avg_price = 0.0
                tot_cost = 0.0
                tot_quantity = 0.0
            if from_date <= date < to_date:
                insert_date: pd.Timestamp = date.normalize()
                data.loc[insert_date, col_purchase] = avg_price
                data.loc[insert_date, col_quantity] = tot_quantity

        if col_purchase not in data.columns:
            return self  # No items inserted.

        # The columns 'purchase' and 'quantity' are forward filled,
        # but only where 'quantity' is > 0
        data[col_quantity] = data[col_quantity].ffill()
        data[col_purchase] = data[col_purchase].ffill()
        mask: pd.Series = data[col_quantity] > 0.0
        data.loc[mask != True, [col_quantity, col_purchase]] = pd.NA
        if data[col_purchase].dropna().empty:
            return self  # There is no actual purchasing data left

        return self._replace(
            operations=operations,
            data=data
        )

    def compute_enter_prices(self) -> t.Self:
        if self.data.empty:
            return self
        data: pd.DataFrame = self.data.copy()
        price: pd.Series = data[Column.PRICE.value]
        sma_fast: pd.Series = data[Column.SMA_FAST.value]
        sma_slow: pd.Series = data[Column.SMA_SLOW.value]

        # Generate the serie of reference prices to compute
        # investment enter prices:
        #  - take the max of the 2 SMA series, where available.
        #  - take the market price as fall-back strategy.
        reference_price: pd.Series = (
            sma_fast
            .combine(sma_slow, max)
            .combine_first(price).dropna()
        )

        # Goes X% down from the reference price at each step:
        last_enter_prices: t.Sequence[float] = []
        x: int
        for x in range(1, 4):
            k: float = self.invest_ratio ** (-x)
            enter: pd.Series = (k * reference_price).dropna()
            enter_col: str = f"{Column.ENTER.value} #{x}"
            data[enter_col] = enter
            last_enter_prices.append(enter.iloc[-1])

        return self._replace(
            data=data,
            last_enter_prices=last_enter_prices,
        )
    
    def compute_exit_prices(self) -> t.Self:
        if self.data.empty:
            return self
        data: pd.DataFrame = self.data.copy()
        price: pd.Series = data[Column.PRICE.value]
        sma: pd.Series = data[Column.SMA_FAST.value]

        # The reference price is the SMA when available,
        # Otherwise the market price:
        price = sma.combine_first(price).dropna()

        # The reference price is the avg. purchase price when
        # available, otherwise the market price:
        if Column.PURCHASE.value in data.columns:
            purchase: pd.Series = data[Column.PURCHASE.value]
            price = purchase.combine_first(price).dropna()

        last_exit_prices: t.Sequence[float] = []
        x: int
        for x in range(1, 4):
            k: float = self.invest_ratio ** x
            exit: pd.Series = (k * price).dropna()
            exit_col: str = f"{Column.EXIT.value} #{x}"
            data[exit_col] = exit
            last_exit_prices.append(exit.iloc[-1])

        return self._replace(
            data=data,
            last_exit_prices=last_exit_prices,
        )

    def print_last_prices(self) -> t.Self:
        last_price: float = pd.NA
        price_col: str
        for price_col in (
            Column.PURCHASE.value, 
            Column.SMA_FAST.value, 
            Column.PRICE.value
        ):
            if  price_col in self.data.columns:
                last_price = self.data[price_col].iloc[-1]
            if not pd.isna(last_price):
                break
        if pd.isna(last_price):
            return self

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
                    return f"{price:.2f}"
                if pos > 0:
                    return f"{Fore.YELLOW}{price:,.2f}{Style.RESET_ALL}"
            return str(price)
        
        prices: t.Sequence[str] = [
            *(
                _represent_price(-1, price) 
                for price in sorted(self.last_enter_prices)
            ),
            _represent_price(0, last_price),
            *(
                _represent_price(1, price) 
                for price in sorted(self.last_exit_prices)
            ),
        ]
        print("Prices:", " | ".join(prices))

        return self

    def print_scores(self) -> t.Self:
        if self.data.empty:
            return self

        data: pd.DataFrame = self.data
        last_price: float = data[Column.PRICE.value].dropna().iloc[-1]

        # With an invest ration of 1.05, the target would be 5%:
        target: float = 100 * (self.invest_ratio - 1.0 )

        # Compute buy score:
        if Column.SMA_FAST.value in data.columns:
            last_sma: float = data[Column.SMA_FAST.value].dropna().iloc[-1]
            score: float = 100 * ((last_sma - last_price) / last_price)
            color: str = Fore.GREEN if score >= target else Fore.LIGHTWHITE_EX
            print(f"{color}Buy score: {score:.2f}%")

        # Compute sell score:
        if Column.PURCHASE.value in data.columns:
            last_purchase: float = self.data[Column.PURCHASE.value].iloc[-1]
            if not pd.isna(last_purchase):
                score: float = 100 * (last_price - last_purchase) / last_purchase
                color = Fore.YELLOW if score >= target else Fore.LIGHTWHITE_EX
                print(f"{color}Sell score: {score:.2f}%")

        return self

    def plot(self) -> t.Self:
        if self.data.empty:
            return self
        
        data: pd.DataFrame = self.data
        first_date: pd.Timestamp = data.index[0]
        last_date: pd.Timestamp = data.index[-1]

        plt.figure(figsize=(12,6))

        # Market prices:
        price: pd.Series = data[Column.PRICE.value]
        last_price: float = price.iloc[-1]
        plt.plot(
            data.index, price, 
            label=f"Market price: {last_price:.2f}",
            linewidth=2, color="green",
        )
        plt.scatter(
            [last_date], [last_price],
            s=80, zorder=3, color="green",
            alpha=0.75, marker=">"
        )
        plt.annotate(
            f"{last_price:.2f}", 
            (last_date, last_price), 
            xytext=(10,-5),
            textcoords="offset points", 
            color="green",
        )

        # SMAs:
        alpha: float
        color: str
        column: str
        linestyle: str
        sma_lenght: int
        for column, sma_lenght, color, linestyle, alpha in [
            (Column.SMA_FAST.value, self.sma_fast_lenght, "orange", "solid", 1.0),
            (Column.SMA_SLOW.value, self.sma_slow_lenght, "red", "dotted", 0.75),
        ]:
            if not column in data.columns:
                continue
            sma: pd.Series = data[column]
            last_sma: float = sma.iloc[-1]
            plt.plot(
                data.index, sma, 
                label=f"SMA-{sma_lenght}: {last_sma:.2f}",
                alpha=alpha, color=color, 
                linewidth=2, linestyle=linestyle,
            )
            if linestyle == "solid":
                plt.scatter(
                    [last_date], [last_sma],
                    s=80, zorder=3, color=color,
                    alpha=0.75 * alpha, marker=">"
                )
                plt.annotate(
                    f"{last_sma:.2f}", 
                    (last_date, last_sma), 
                    xytext=(10,-5), 
                    textcoords="offset points", 
                    color=color,
                )

        # Plot investment activities:
        purchase_col: str = Column.PURCHASE.value
        if purchase_col in data.columns:
            purchases: pd.Series = data[Column.PURCHASE.value].dropna()
            last_valid_purchase: float = float(
                data[Column.PURCHASE.value].dropna().iloc[-1]
            )
            plt.plot(
                data.index, data[purchase_col], 
                label=f"Avg. purch. price: {last_valid_purchase:.2f}",
                linewidth=2, color="blue",
            )
            last_purchase: float = float(
                data[Column.PURCHASE.value].iloc[-1]
            )
            if not pd.isna(last_purchase):
                plt.scatter(
                    [last_date], [last_purchase],
                    s=80, zorder=3, color="blue",
                    alpha=0.75, marker=">"
                )
                plt.annotate(
                    f"{last_purchase:.2f}", 
                    (last_date, last_purchase), 
                    xytext=(10,-5), 
                    textcoords="offset points", 
                    color="blue",
                )
        if not self.operations.empty:
            for date, price, quantity in zip(
                self.operations.index, 
                self.operations[Column.PRICE.value],
                self.operations[Column.QUANTITY.value],
            ):
                if first_date <= date <= last_date: 
                    plt.scatter(
                        [date], [price],
                        s=80, zorder=3, 
                        color="lightgreen" if quantity > 0 else "red",
                        marker="o",
                        alpha=0.5,
                    )

        # Enter/exit prices:
        if self.last_enter_prices:
            for i, price in enumerate(self.last_enter_prices):
                self.plot_price_level(
                    color="gray",
                    column=f"{Column.ENTER.value} #{i + 1}",
                    price=price,
                )
        if self.last_exit_prices:
            for i, price in enumerate(self.last_exit_prices):
                self.plot_price_level(
                    color="gray",
                    column=f"{Column.EXIT.value} #{i + 1}",
                    price=price,
                )

        ticker: yf.Ticker = self.ticker or yf.Ticker(self.security)
        currency: str = ticker.info.get("currency") or "USD"

        plt.title(
            f"{self.security} ({self.company_name}) - "
            f"Market price with SMA({self.sma_fast_lenght})"
        )
        plt.xlabel("Date")
        plt.ylabel(f"Price ({currency})")
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
        if column in self.data.columns:
            plt.plot(
                self.data.index,
                self.data[column],
                label=f"{column}: {price:.2f}",
                linewidth=1,
                color=color,
                linestyle="--",
            )

    def show_company_table(self) -> t.Self:
        ticker:yf.Ticker = self.ticker or yf.Ticker(self.security)
        info = dict(ticker.info)

        revenue: int | None = info.get("totalRevenue", None)
        if revenue is not None:
            info["totalRevenue"] = "{:,d}".format(revenue)

        gross_profits: int | None = info.get("grossProfits", None)
        if gross_profits is not None:
            info["grossProfits"] = "{:,d}".format(gross_profits)

        data = {
            "Company Name": [self.company_name],
            "Current Price": [info.get("currentPrice")],
            "Forward EPS": [info.get("forwardEps")],
            "Gross Profit (ttm)": [info.get("grossProfits")],
            "Industry": [info.get("industry")],
            "Revenue (ttm)": [info.get("totalRevenue")],
            "Target High Price": [info.get("targetHighPrice")],
            "Target Low Price": [info.get("targetLowPrice")],
            "Target Mean Price": [info.get("targetMeanPrice")],
            "Trailing EPS": [info.get("trailingEps")],
            "Website": [info.get("website")],
        }
        df = pd.DataFrame(
            data=data,
            index=[self.security],
        )
        display(df.T)
        return self

    def show_status(self) -> t.Self:
        ticker:yf.Ticker = self.ticker or yf.Ticker(self.security)
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


def run(
        security: str, 
        **kwargs
):
    (
        Context(
            security=security,
            **kwargs
        ).load()
        .append_last_price()
        .compute_SMAs()
        .import_trades()
        .compute_enter_prices()
        .compute_exit_prices()
        .print_last_prices()
        .print_scores()
        .plot()
        .show_company_table()
        .show_status()
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
