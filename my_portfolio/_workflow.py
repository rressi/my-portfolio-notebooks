from colorama import Fore, Style
import enum
import io
from pathlib import Path
import typing as t

from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from my_portfolio._currency import (
    to_currency as convert_currency,
)
from my_portfolio._import_trades import (
    import_many_trades,
)
from my_portfolio._numerics import (
    first_non_na,
)


class Column(enum.StrEnum):
    BALANCE = "balance"
    CUM_QUANTITY = "cum-quantity"
    DATE = "date"
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
    last_date: pd.Timestamp | None = None
    last_enter_prices: t.Sequence[float] = tuple()
    last_exit_prices: t.Sequence[float] = tuple()
    last_price: float = pd.NA
    market_date: pd.Timestamp | None = None
    market_price: float = pd.NA
    operations: pd.DataFrame = pd.DataFrame()
    purchase_date: pd.Timestamp | None = None
    purchase_price: float = pd.NA
    sma_fast_lenght: int = 10
    sma_slow_lenght: int = 20
    start_date: str | pd.Timestamp = "2025-06-01"
    trades_isin: str | None = None
    ticker: yf.Ticker | None = None

    def load(self) -> t.Self:
        ticker: yf.Ticker = yf.Ticker(self.security)
        company_name: str = get_company_name(ticker)
        tz_name = ticker.info.get("exchangeTimezoneName")

        start_date: pd.Timestamp = pd.Timestamp(self.start_date).tz_localize(tz_name)
        start_date_before: pd.Timestamp = min(
            start_date - pd.Timedelta(days=self.sma_fast_lenght + 1),
            start_date - pd.Timedelta(days=self.sma_slow_lenght + 1),
        )
        data: pd.DataFrame = (
            ticker.history(
                start=start_date_before,
                interval="1d",
            )[["Close"]]
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

    def handle_last_price(self) -> t.Self:
        if self.data.empty:
            return self

        market_hourly: pd.DataFrame = self.ticker.history(
            period="4d",
            interval="1h",
            prepost=False,
            auto_adjust=False,
        ).dropna(
            subset=["Close"],
        )
        if market_hourly.empty:
            last_market_date: pd.Timestamp = self.data.index[-1]
            last_market_price: float = self.data[Column.PRICE.value].iloc[-1]
            return self._replace(
                market_date=last_market_date,
                market_price=last_market_price,
            )

        last_market_date: pd.Timestamp = market_hourly.index[-1]
        last_market_price: float = self.data[Column.PRICE.value].iloc[-1]

        prepost_hourly: pd.DataFrame = self.ticker.history(
            period="4d",
            interval="1h",
            prepost=True,
            auto_adjust=False,
        ).dropna(
            subset=["Close"],
        )
        if prepost_hourly.empty:
            return self._replace(
                market_date=last_market_date,
                market_price=last_market_price,
            )

        last_date: pd.Timestamp = last_market_date
        last_price: float = last_market_price

        last_prepos_date: pd.Timestamp = prepost_hourly.index[-1]
        last_prepos_price: float = prepost_hourly["Close"].iloc[-1]
        if last_prepos_date > last_market_date and not pd.isna(last_prepos_price):
            last_date = last_prepos_date.normalize() + pd.Timedelta(days=1)
            last_price = last_prepos_price

        return self._replace(
            market_date=last_market_date.normalize(),
            market_price=last_market_price,
            last_date=last_date,
            last_price=last_price,
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
            sna_fast.first_valid_index() or -1, sna_slow.first_valid_index() or -1
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
            data_folder=Path("data"), sql_path=Path("data/import.sql")
        )

        if self.trades_isin is not None:
            operations = operations[operations["ISIN"] == self.trades_isin]
            if operations.empty:
                print(f"No operations found for ISIN '{self.trades_isin}'")
                return self

        else:
            selected_ticker: str = self.security
            match (self.ticker.info.get("quoteType"), self.ticker.info.get("fromCurrency")):
                case ["CRYPTOCURRENCY", crypto_ticker]:
                    selected_ticker = crypto_ticker

            operations = operations[operations["ticker"] == selected_ticker]
            if operations.empty:
                print(f"No operations found for ticker '{selected_ticker}'")
                return self

        # Converts the timestamps
        operations = operations.tz_convert(self.data.index.tz)

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

        col_balance: str = Column.BALANCE.value
        col_cum_quantity: str = Column.CUM_QUANTITY.value
        col_purchase: str = Column.PURCHASE.value
        col_quantity: str = Column.QUANTITY.value

        # Compute average purchase price:
        purchase_price: float = 0.0
        tot_cost: float = 0.0
        tot_quantity: float = 0.0
        purchase_date: pd.Timestamp
        balance: float = 0.0
        row: t.Mapping[str, t.Any]
        for purchase_date, row in operations.iterrows():
            costs: float = row["costs"]
            price: float = row["price"]
            quantity: float = row["quantity"]

            balance -= costs + (price * quantity)
            tot_quantity += quantity

            if tot_quantity <= 0.0:
                purchase_price = 0.0
                tot_cost = 0.0
                tot_quantity = 0.0
            elif quantity < 0.0:
                tot_cost += quantity * purchase_price
            else:
                tot_cost += quantity * price
                purchase_price = tot_cost / tot_quantity

            operations.loc[purchase_date, col_purchase] = purchase_price
            operations.loc[purchase_date, col_cum_quantity] = tot_quantity
            operations.loc[purchase_date, col_balance] = balance

            if from_date <= purchase_date < to_date:
                insert_date: pd.Timestamp = purchase_date.normalize()
                data.loc[insert_date, col_purchase] = purchase_price
                data.loc[insert_date, col_quantity] = tot_quantity

        if col_purchase not in data.columns:
            # Nothing has been inserted into the data frame:
            return self._replace(
                operations=operations,
                purchase_price=purchase_price,
                purchase_date=purchase_date,
            )
        
        # The columns 'purchase' and 'quantity' are forward filled,
        # but only where 'quantity' is > 0
        data[col_quantity] = data[col_quantity].ffill()
        data[col_purchase] = data[col_purchase].ffill()
        mask: pd.Series = data[col_quantity] > 0.0
        data.loc[mask != True, [col_quantity, col_purchase]] = pd.NA
        if data[col_purchase].dropna().empty:
            return self._replace(
                operations=operations,
                purchase_price=purchase_price,
                purchase_date=purchase_date,
            )  # No valid purchase price in the data frame

        return self._replace(
            data=data,
            operations=operations,
            purchase_price=purchase_price,
            purchase_date=purchase_date,
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
            sma_fast.combine(sma_slow, max).combine_first(price).dropna()
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
            k: float = self.invest_ratio**x
            exit: pd.Series = (k * price).dropna()
            exit_col: str = f"{Column.EXIT.value} #{x}"
            data[exit_col] = exit
            last_exit_prices.append(exit.iloc[-1])

        return self._replace(
            data=data,
            last_exit_prices=last_exit_prices,
        )

    def print_last_prices(self) -> t.Self:
        reference_price: float = pd.NA
        for price_candidate in (
            self.purchase_price,
            Column.PURCHASE,
            Column.SMA_FAST,
            Column.SMA_SLOW,
            self.last_price,
            Column.PRICE,
        ):
            match price_candidate:
                case float(price) if not pd.isna(price):
                    reference_price = price
                    break
                case Column(column) if column.value in self.data.columns:
                    price: float = self.data[column.value].iloc[-1]
                    if not pd.isna(price):
                        reference_price = price
                        break
        if pd.isna(reference_price):
            return self

        def _represent_price(
            pos: int,
            price: float | None,
        ) -> str:
            if pd.isna(price):
                return "N/A"
            if isinstance(price, float):
                if pos < 0:
                    return f"{Fore.GREEN}{price:,.2f}{Style.RESET_ALL}"
                if pos == 0:
                    return f"{price:,.2f}"
                if pos > 0:
                    return f"{Fore.YELLOW}{price:,.2f}{Style.RESET_ALL}"
            return str(price)

        prices: t.Sequence[str] = [
            *(_represent_price(-1, price) for price in sorted(self.last_enter_prices)),
            _represent_price(0, reference_price),
            *(_represent_price(1, price) for price in sorted(self.last_exit_prices)),
        ]
        print("Prices:", " | ".join(prices))

        return self

    def print_scores(self) -> t.Self:
        if self.data.empty:
            return self

        data: pd.DataFrame = self.data
        reference_price: float = self.last_price
        if pd.isna(reference_price):
            return self

        # With an invest ration of 1.05, the target would be 5%:
        target_score: float = 100 * (self.invest_ratio - 1.0)

        # Compute buy score:
        if Column.SMA_FAST.value in data.columns:
            last_sma: float = data[Column.SMA_FAST.value].dropna().iloc[-1]
            if not pd.isna(last_sma):
                score: float = 100 * ((last_sma - reference_price) / reference_price)
                color: str = Fore.GREEN if score >= target_score else Fore.LIGHTWHITE_EX
                print(f"{color}Buy score: {score:.2f}%")

        # Compute sell score:
        if Column.PURCHASE.value in data.columns:
            last_purchase: float = self.data[Column.PURCHASE.value].iloc[-1]
            if not pd.isna(last_purchase):
                score: float = 100 * (reference_price - last_purchase) / last_purchase
                color = Fore.YELLOW if score >= target_score else Fore.LIGHTWHITE_EX
                print(f"{color}Sell score: {score:.2f}%")

        return self

    def plot(self) -> t.Self:
        data: pd.DataFrame = self.data
        if data.empty:
            return self

        first_date: pd.Timestamp = data.index[0]
        last_date: pd.Timestamp = first_non_na(
            self.last_date,
            data.index[-1],
        )
        market_price: pd.Series = data[Column.PRICE.value]
        last_market_price: float = first_non_na(
            self.market_price,
            market_price.iloc[-1],
        )

        plt.figure(figsize=(12, 6))

        # Last price:
        annotate_market_price: bool = True
        if not pd.isna(self.last_price) and self.last_price != self.market_price:
            annotate_market_price = False
            plt.plot(
                [self.market_date, self.last_date],  # x-coordinates
                [last_market_price, self.last_price],  # y-coordinates
                color="purple",
                label=f"Last price: {self.last_price:,.2f}",
                linestyle="--",
                linewidth=1,
            )
            plt.scatter(
                [self.last_date],
                [self.last_price],
                alpha=0.75,
                color="purple",
                marker=">",
                s=80,
                zorder=3,
            )
            plt.annotate(
                f"{self.last_price:,.2f}",
                (self.last_date, self.last_price),
                color="purple",
                textcoords="offset points",
                xytext=(10, -5),
                zorder=10,
            )

        # Market prices:
        if not market_price.dropna().empty:
            plt.plot(
                data.index,
                market_price,
                color="green",
                label=f"Market price: {last_market_price:,.2f}",
                linewidth=2,
            )
            if annotate_market_price:
                plt.scatter(
                    [last_date],
                    [last_market_price],
                    alpha=0.75,
                    color="green",
                    marker=">",
                    s=80,
                    zorder=3,
                )
                plt.annotate(
                    f"{last_market_price:,.2f}",
                    (last_date, last_market_price),
                    color="green",
                    textcoords="offset points",
                    xytext=(10, -5),
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
                data.index,
                sma,
                label=f"SMA-{sma_lenght}: {last_sma:.2f}",
                alpha=alpha,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )
            if linestyle == "solid":
                plt.scatter(
                    [last_date],
                    [last_sma],
                    s=80,
                    zorder=3,
                    color=color,
                    alpha=0.75 * alpha,
                    marker=">",
                )
                plt.annotate(
                    f"{last_sma:.2f}",
                    (last_date, last_sma),
                    xytext=(10, -5),
                    textcoords="offset points",
                    color=color,
                )

        # Plot investment activities:
        purchase_col: str = Column.PURCHASE.value
        if purchase_col in data.columns:
            purchases: pd.Series = data[Column.PURCHASE.value]
            last_valid_purchase: float = float(purchases.dropna().iloc[-1])
            plt.plot(
                data.index,
                data[purchase_col],
                label=f"Avg. purch. price: {last_valid_purchase:.2f}",
                linewidth=2,
                color="blue",
            )
            last_purchase: float = float(purchases.iloc[-1])
            if not pd.isna(last_purchase):
                plt.scatter(
                    [last_date],
                    [last_purchase],
                    s=80,
                    zorder=3,
                    color="blue",
                    alpha=0.75,
                    marker=">",
                )
                plt.annotate(
                    f"{last_purchase:.2f}",
                    (last_date, last_purchase),
                    xytext=(10, -5),
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
                        [date],
                        [price],
                        s=80,
                        zorder=3,
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
        ticker: yf.Ticker = self.ticker or yf.Ticker(self.security)
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

    def show_operations(self) -> t.Self:
        if self.operations.empty:
            print("No operations found.")
            return self

        col_price: str = Column.PRICE.value
        col_quantity: str = Column.QUANTITY.value
        report_column: list[str] = [
            "At",
            "Unit price",
            "Quantity",
        ]

        last_ops = self.operations.tail(10)
        min_quantity = first_non_na(last_ops[col_quantity].min(), 1.0)

        def _format_date(date: pd.Timestamp) -> str:
            return date.strftime("%Y-%m-%d %H:%M")

        def _format_price(price: float) -> str:
            return f"{price:,.2f}"

        def _format_quantity(quantity: float) -> str:
            if min_quantity >= 1:
                return f"{quantity:,.0f}"
            if min_quantity >= 0.1:
                return f"{quantity:,.3f}"
            else:
                return f"{quantity:,.6f}"

        report: pd.DataFrame = pd.DataFrame()
        report.index = last_ops.index
        report["Time"] = last_ops.index.map(_format_date)
        report["Quantity"] = last_ops[col_quantity].map(_format_quantity)
        report["Price"] = last_ops[col_price].map(_format_price)
        report["Tot"] = (last_ops[col_quantity] * last_ops[col_price]).map(
            _format_price
        )

        col_cum_quantity: str = Column.CUM_QUANTITY.value
        if col_cum_quantity in last_ops.columns:
            report["Wallet"] = last_ops[col_cum_quantity].map(_format_quantity)
            report_column.append(col_cum_quantity)

        col_purhcase: str = Column.PURCHASE.value
        if col_purhcase in last_ops.columns:
            report["Avg. price"] = last_ops[col_purhcase].map(_format_price)
            report_column.append(col_purhcase)

        col_balance: str = Column.BALANCE.value
        if col_balance in last_ops.columns and col_cum_quantity in last_ops.columns:
            report["Balance"] = (
                (last_ops[col_price] * last_ops[col_cum_quantity])
                + last_ops[col_balance]
            ).map(_format_price)
            report_column.append(col_balance)

            last_price: float = pd.NA
            last_date: pd.Timestamp | None = None
            last_quantity: float = last_ops[col_cum_quantity].iloc[-1]
            if last_quantity > 0.0:
                last_date = first_non_na(self.last_date, self.market_date)
                last_price = first_non_na(self.last_price, self.market_price)

            # Add a summary line with the last price:
            if last_date is not None and not pd.isna(last_price):
                last_balance: float = last_ops[col_balance].iloc[-1] + (
                    last_price * last_quantity
                )
                report.loc[last_date, "Time"] = _format_date(last_date)
                report.loc[last_date, "Quantity"] = _format_quantity(last_quantity)
                report.loc[last_date, "Price"] = _format_price(last_price)
                report.loc[last_date, "Tot"] = _format_price(last_price * last_quantity)
                report.loc[last_date, "Wallet"] = _format_quantity(last_quantity)
                report.loc[last_date, "Balance"] = _format_price(last_balance)

        report = report.set_index("Time")
        display(report)
        return self

    def show_status(self) -> t.Self:
        ticker: yf.Ticker = self.ticker or yf.Ticker(self.security)
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
            _get_field(key): [_get_value(key)] for key in fast_info.keys()
        }
        df = pd.DataFrame(
            data=data,
            index=[self.security],
        )
        display(df.T)
        return self


def run(security: str, **kwargs):
    (
        Context(security=security, **kwargs)
        .load()
        .handle_last_price()
        .compute_SMAs()
        .import_trades()
        .compute_enter_prices()
        .compute_exit_prices()
        .print_last_prices()
        .print_scores()
        .plot()
        .show_operations()
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
