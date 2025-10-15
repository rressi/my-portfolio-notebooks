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

pd.set_option("display.max_rows", 1000)


class Column(enum.StrEnum):
    BALANCE = "balance"
    COSTS = "costs"
    CUM_QUANTITY = "cum-quantity"
    DATE = "date"
    ENTER = "enter"
    EXIT = "exit"
    INVESTMENT_BALANCE = "invest-balance"
    INVESTMENT_COST = "invest-cost"
    PRICE = "price"  # Market price
    PURCHASE = "purchase"  # Avg. purchase price
    QUANTITY = "quantity"
    SMA_5 = "sma-10"
    SMA_50 = "sma-50"
    SMA_150 = "sma-150"


class Context(t.NamedTuple):
    security: str

    company_name: str = ""
    currency: str | None = None
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
    sma_5: int = 5
    sma_50: int = 50
    sma_150: int = 150
    trades_isin: str | None = None
    ticker: yf.Ticker | None = None

    def load(self) -> t.Self:
        ticker: yf.Ticker = yf.Ticker(self.security)
        company_name: str = get_company_name(ticker)

        # Get the hourly data for the last month:
        data: pd.DataFrame = (
            ticker.history(
                period="1mo",
                interval="1h",
            )[["Close"]]
            .rename(columns={"Close": Column.PRICE.value})
            .sort_index()
        )
        if data.empty:
            raise ValueError(f"No data found for ticker: '{self.security}'")

        currency: str = self.currency or ticker.info.get("currency")
        if currency is None:
            raise ValueError(f"No currency found for ticker: '{self.security}'")

        return self._replace(
            company_name=company_name,
            currency=currency,
            data=data,
            ticker=ticker,
        )

    def handle_last_price(self) -> t.Self:
        if self.data.empty:
            return self

        # Get the 5 days market prices with 1-minute interval:
        col_price: str = Column.PRICE.value
        market_1m: pd.DataFrame = (
            self.ticker.history(
                period="5d",
                interval="1m",
                prepost=False,
                auto_adjust=False,
            )[["Close"]]
            .rename(columns={"Close": col_price})
            .dropna(subset=[col_price])
        )

        # If no market data is available, at 1-minute interval, 
        # then we use the last price:
        if market_1m.empty:
            last_market_date: pd.Timestamp = self.data.index[-1]
            last_market_price: float = self.data[col_price].iloc[-1]
            return self._replace(
                market_date=last_market_date,
                market_price=last_market_price,
            )

        # The last market price is the last price at 1-minute interval:
        last_market_date: pd.Timestamp = market_1m.index[-1]
        last_market_price: float = market_1m[col_price].iloc[-1]

        # Get the 5 days prices, with prepros, with 1-minute interval:
        prepost_1m: pd.DataFrame = (
            self.ticker.history(
                period="5d",
                interval="1m",
                prepost=True,
                auto_adjust=False,
            )
            .rename(columns={"Close": col_price})
            .dropna(subset=[col_price])
        )
        if prepost_1m.empty:
            # No pre/post market data available: 
            # - just return with what we have:
            return self._replace(
                market_date=last_market_date,
                market_price=last_market_price,
            )

        last_date: pd.Timestamp = last_market_date
        last_price: float = last_market_price

        last_prepos_date: pd.Timestamp = prepost_1m.index[-1]
        last_prepos_price: float = prepost_1m[col_price].iloc[-1]
        if (
            not pd.isna(last_prepos_price) 
            and last_prepos_date > last_market_date
        ):
            last_date = last_prepos_date.normalize() + pd.Timedelta(days=1)
            last_price = last_prepos_price

        return self._replace(
            market_date=last_market_date.normalize(),
            market_price=last_market_price,
            last_date=last_date,
            last_price=last_price,
        )

    def compute_SMAs(self) -> t.Self:
        if self.data.empty or self.ticker is None:
            return self

        col_price: str = Column.PRICE.value

        # Get the date interval to insert only operations
        # within the range of our market data:
        data: pd.DataFrame = self.data.copy()
        from_date: pd.Timestamp = (
            # first date moved to the midnight before
            data.index[0].normalize()
        )
        to_date: pd.Timestamp = (
            # last date moved to the midnight before
            data.index[-1].normalize()
            # but of the next day
            + pd.Timedelta(days=1)
        )

        # Compute SMAs and insert them into the data frame:
        for sma_col, sma_lenght, period in zip(
            [Column.SMA_5.value, Column.SMA_50.value, Column.SMA_150.value],
            [self.sma_5, self.sma_50, self.sma_150],
            ["3mo", "6mo", "1y"],
        ):
            sma_x: pd.Series = (
                self.ticker.history(period=period, interval="1d")
                [["Close"]] # Get only the 'Close' column
                .sort_index() # Sort by date (index)
                ["Close"] # Get the 'Close' series
                .rolling(sma_lenght) # Compute the rolling window
                .mean() # Compute the mean over the window
                .dropna() # Drop NaN values
            )
            if sma_x.empty:
                print(
                    f"No daily data found for {self.security}, "
                    f"cannot compute SMA-{sma_lenght}"
                )
                continue

            ts: pd.Timestamp
            sma_value: float
            for ts, sma_value in sma_x.items():
                if from_date <= ts < to_date:
                    insertion_ts: pd.Timestamp = get_insertion_ts(
                        data=data, 
                        original_ts=ts,
                    )
                    data.loc[insertion_ts, sma_col] = sma_value

        # Fordard fill the SMAs:
        for sma_col in [Column.SMA_5.value, Column.SMA_50.value, Column.SMA_150.value]:
           if sma_col in data.columns:
               data[sma_col] = data[sma_col].ffill()

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

        # Converts the timestamps of the operations to the timezone of the ticker:
        operations = operations.tz_convert(self.data.index.tz)

        # Convert the currency of the operations to the currency of the ticker:
        if "currency" in operations.columns:
            operations = convert_currency(operations, self.currency)

        # Get the date interval to insert only operations
        # within the range of our market data:
        data: pd.DataFrame = self.data.copy()
        from_date: pd.Timestamp = (
            # first date moved to the midnight before
            data.index[0].normalize()
        )
        to_date: pd.Timestamp = (
            # last date moved to the midnight before
            data.index[-1].normalize()
            # but of the next day
            + pd.Timedelta(days=1)
        )

        col_balance: str = Column.BALANCE.value
        col_costs: str = Column.COSTS.value
        col_cum_quantity: str = Column.CUM_QUANTITY.value
        col_investment_balance: str = Column.INVESTMENT_BALANCE.value
        col_investment_cost: str = Column.INVESTMENT_COST.value
        col_price: str = Column.PRICE.value
        col_purchase: str = Column.PURCHASE.value
        col_quantity: str = Column.QUANTITY.value

        # Compute balance and average purchase price:
        purchase_price: float = 0.0
        tot_cost: float = 0.0
        tot_quantity: float = 0.0
        op_ts: pd.Timestamp
        balance: float = 0.0
        inv_balance: float = 0.0
        inv_cost: float = 0.0
        row: t.Mapping[str, t.Any]
        for op_ts, row in operations.iterrows():
            costs: float = row[col_costs]
            price: float = row[col_price]
            quantity: float = row[col_quantity]

            # Update balance and total quantity:
            balance -= costs + (price * quantity)
            tot_quantity += quantity

            if tot_quantity <= 0.0:
                # We sold all the stock we had in the wallet:
                purchase_price = 0.0
                tot_cost = 0.0
                tot_quantity = 0.0
                inv_cost = 0.0
                inv_balance = 0.0
            elif quantity < 0.0:
                # We sold part of the stock we had in the wallet:
                # - The total cost is reduced
                # - The average purchase price remains the same
                inv_balance -= costs + (price * quantity)
                inv_cost += costs
                tot_cost += quantity * purchase_price
            elif quantity > 0.0:
                # We bought more stock:
                # - The total cost is increased
                # - The average purchase price is updated
                inv_balance -= costs + (price * quantity)
                inv_cost += costs
                tot_cost += quantity * price
                purchase_price = tot_cost / tot_quantity
            else: # quantity == 0.0
                # No stock bought or sold, just costs:
                inv_balance -= costs
                inv_cost += costs

            # Add datapoing on new columns: 'purchase', 'cum-quantity', 'balance':
            operations.loc[op_ts, col_purchase] = purchase_price
            operations.loc[op_ts, col_cum_quantity] = tot_quantity
            operations.loc[op_ts, col_balance] = balance
            operations.loc[op_ts, col_investment_balance] = inv_balance
            operations.loc[op_ts, col_investment_cost] = inv_cost

            # Insert datapoint in the data frame only if within the date range:
            if from_date <= op_ts < to_date:
                insertion_ts: pd.Timestamp = get_insertion_ts(
                    data=data, 
                    original_ts=op_ts,
                )
                data.loc[insertion_ts, col_purchase] = purchase_price
                data.loc[insertion_ts, col_quantity] = tot_quantity

        # If we don't have any stock in the wallet, then there is no
        # valid average purchase price anymore:
        if tot_quantity <= 0.0:
            purchase_price = pd.NA
            op_ts = None

        if col_purchase not in data.columns:
            # Nothing has been inserted into the data frame:
            return self._replace(
                operations=operations,
                purchase_price=purchase_price,
                purchase_date=op_ts,
            )
        
        # The columns 'purchase' and 'quantity' are forward filled,
        # but only where 'quantity' is > 0
        data[col_quantity] = data[col_quantity].ffill()
        data[col_purchase] = data[col_purchase].ffill()

        # When we don't have any stock in the wallet, then there we put NaN
        # on the columns 'purchase' and 'quantity':
        mask: pd.Series = data[col_quantity] > 0.0
        data.loc[mask == False, [col_quantity, col_purchase]] = pd.NA
        if data[col_purchase].dropna().empty:
            # No valid purchase price in the data frame:
            # - We don't update the data frame
            return self._replace(
                operations=operations,
                purchase_price=purchase_price,
                purchase_date=op_ts,
            )

        return self._replace(
            data=data,
            operations=operations,
            purchase_price=purchase_price,
            purchase_date=op_ts,
        )

    def compute_enter_prices(self) -> t.Self:
        if self.data.empty:
            return self
        data: pd.DataFrame = self.data.copy()
        price: pd.Series = data[Column.PRICE.value]
        sma_5: pd.Series = data[Column.SMA_5.value]
        sma_50: pd.Series = data[Column.SMA_50.value]

        # Generate the serie of reference prices to compute
        # investment enter prices:
        #  - take the max of the 2 SMA series, where available.
        #  - take the market price as fall-back strategy.
        reference_price: pd.Series = (
            sma_5.combine(sma_50, max).combine_first(price).dropna()
        )

        # Goes X% down from the reference price at each step:
        last_enter_prices: t.Sequence[float] = []
        x: int
        for x in range(1, 4):
            k: float = self.invest_ratio ** (-x)
            enter_price_k: pd.Series = (k * reference_price).dropna()
            enter_col_k: str = f"{Column.ENTER.value} #{x}"
            data[enter_col_k] = enter_price_k
            last_enter_prices.append(enter_price_k.iloc[-1])

        return self._replace(
            data=data,
            last_enter_prices=last_enter_prices,
        )

    def compute_exit_prices(self) -> t.Self:
        if self.data.empty:
            return self
        data: pd.DataFrame = self.data.copy()
        price: pd.Series = data[Column.PRICE.value]
        sma_5: pd.Series = data[Column.SMA_5.value]

        # The reference price is the SMA when available,
        # Otherwise the market price:
        reference_price = sma_5.combine_first(price).dropna()

        # The reference price is the avg. purchase price when
        # available, otherwise the market price:
        if Column.PURCHASE.value in data.columns:
            purchase: pd.Series = data[Column.PURCHASE.value]
            reference_price = purchase.combine_first(reference_price).dropna()

        last_exit_prices: t.Sequence[float] = []
        x: int
        for x in range(1, 4):
            k: float = self.invest_ratio**x
            exit_price_k: pd.Series = (k * reference_price).dropna()
            exit_col_k: str = f"{Column.EXIT.value} #{x}"
            data[exit_col_k] = exit_price_k
            last_exit_prices.append(exit_price_k.iloc[-1])

        return self._replace(
            data=data,
            last_exit_prices=last_exit_prices,
        )

    def print_last_prices(self) -> t.Self:
        ref_price: float = pd.NA
        for price_candidate in (
            self.purchase_price,
            Column.PURCHASE,
            Column.SMA_5,
            Column.SMA_50,
            Column.SMA_150,
            self.last_price,
            Column.PRICE,
        ):
            match price_candidate:
                case float(price) if not pd.isna(price):
                    ref_price = price
                    break
                case Column(column) if column.value in self.data.columns:
                    price: float = self.data[column.value].iloc[-1]
                    if not pd.isna(price):
                        ref_price = price
                        break
        if pd.isna(ref_price):
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
            _represent_price(0, ref_price),
            *(_represent_price(1, price) for price in sorted(self.last_exit_prices)),
        ]
        print("Prices:", " | ".join(prices))

        return self

    def print_scores(self) -> t.Self:
        if self.data.empty:
            return self

        data: pd.DataFrame = self.data
        ref_price: float = first_non_na(
            self.last_price, 
            self.market_price,
        )
        if pd.isna(ref_price):
            return self

        # With an invest ration of 1.05, the target would be 5%:
        target_score: float = 100 * (self.invest_ratio - 1.0)

        # Compute buy score:
        if Column.SMA_5.value in data.columns:
            last_sma: float = data[Column.SMA_5.value].dropna().iloc[-1]
            if not pd.isna(last_sma):
                score: float = 100 * ((last_sma - ref_price) / ref_price)
                color: str = Fore.GREEN if score >= target_score else Style.RESET_ALL
                print(f"{color}Buy score: {score:.2f}%")

        # Compute sell score:
        if Column.PURCHASE.value in data.columns:
            last_purchase: float = self.data[Column.PURCHASE.value].iloc[-1]
            if not pd.isna(last_purchase):
                score: float = 100 * (ref_price - last_purchase) / last_purchase
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

        plt.figure(figsize=(12, 9))

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
            (Column.SMA_5.value, self.sma_5, "orange", "solid", 1.0),
            (Column.SMA_50.value, self.sma_50, "red", "dotted", 0.75),
            (Column.SMA_150.value, self.sma_150, "darkred", "dotted", 0.75),
        ]:
            if not column in data.columns:
                continue

            sma: pd.Series = data[column]
            last_sma: float = sma.iloc[-1]
            plt.plot(
                data.index,
                sma,
                label=f"SMA-{sma_lenght}: {last_sma:,.2f}",
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
                    f"{last_sma:,.2f}",
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
                label=f"Avg. purch. price: {last_valid_purchase:,.2f}",
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
                    f"{last_purchase:,.2f}",
                    (last_date, last_purchase),
                    xytext=(10, -5),
                    textcoords="offset points",
                    color="blue",
                )
        if not self.operations.empty:
            date: pd.Timestamp
            price: float
            quantity: float
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
                        alpha=0.666,
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

        plt.title(f"{self.security} ({self.company_name})")
        plt.xlabel("Date")
        plt.ylabel(f"Price ({self.currency})")
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

        last_ops = self.operations
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

        value: pd.Series = last_ops[col_quantity] * last_ops[col_price]
        report["Value"] = value.map(_format_price)

        col_cum_quantity: str = Column.CUM_QUANTITY.value
        if col_cum_quantity in last_ops.columns:
            report["Wallet"] = last_ops[col_cum_quantity].map(_format_quantity)

        col_purhcase: str = Column.PURCHASE.value
        if col_purhcase in last_ops.columns:
            report["Avg. price"] = last_ops[col_purhcase].map(_format_price)

        col_balance: str = Column.BALANCE.value
        col_inv_balance: str = Column.INVESTMENT_BALANCE.value
        col_inv_cost: str = Column.INVESTMENT_COST.value
        if col_balance in last_ops.columns and col_cum_quantity in last_ops.columns:
            report["Inv. Cost"] = last_ops[col_inv_cost].map(_format_price)

            report["Inv. Balance"] = (
                (last_ops[col_price] * last_ops[col_cum_quantity])
                + last_ops[col_inv_balance]
            ).map(_format_price)

            report["Balance"] = (
                (last_ops[col_price] * last_ops[col_cum_quantity])
                + last_ops[col_balance]
            ).map(_format_price)

            last_price: float = pd.NA
            last_date: pd.Timestamp | None = None
            last_quantity: float = last_ops[col_cum_quantity].iloc[-1]
            if last_quantity > 0.0:
                last_date = first_non_na(self.last_date, self.market_date)
                last_price = first_non_na(self.last_price, self.market_price)

            # Add a summary line with the last price:
            if last_date is not None and not pd.isna(last_price):
                last_value: float = last_price * last_quantity
                last_inv_balance: float = last_ops[col_inv_balance].iloc[-1]
                last_inv_cost: float = last_ops[col_inv_cost].iloc[-1]
                last_balance: float = last_ops[col_balance].iloc[-1]

                report.loc[last_date, "Time"] = _format_date(last_date)
                report.loc[last_date, "Quantity"] = _format_quantity(last_quantity)
                report.loc[last_date, "Price"] = _format_price(last_price)
                report.loc[last_date, "Value"] = _format_price(last_value)
                report.loc[last_date, "Wallet"] = _format_quantity(last_quantity)
                report.loc[last_date, "Inv. Cost"] = _format_price(last_inv_cost)
                report.loc[last_date, "Inv. Balance"] = _format_price(last_value + last_inv_balance)
                report.loc[last_date, "Balance"] = _format_price(last_value + last_balance)

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

def get_insertion_ts(
        data: pd.DataFrame,
        original_ts: pd.Timestamp,
) -> pd.Timestamp:
    # Work on a sorted *view* of the index; this does not modify `data`
    idx_sorted: pd.Index = data.index.sort_values()

    if len(idx_sorted) == 0:
        raise ValueError("Cannot snap: DataFrame index is empty.")

    # Find insertion point
    pos: int = idx_sorted.searchsorted(original_ts)

    # Pick nearest neighbor without assuming any interval
    nearest_ts: pd.Timestamp
    if pos == 0:
        nearest_ts = idx_sorted[0]
    elif pos == len(idx_sorted):
        nearest_ts = idx_sorted[-1]
    else:
        before = idx_sorted[pos - 1]
        after  = idx_sorted[pos]
        # choose the closer one; ties go to 'before'
        nearest_ts = (
            before 
            if (original_ts - before) <= (after - original_ts) 
            else after
        )

    return nearest_ts
