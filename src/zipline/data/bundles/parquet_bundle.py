import logging
import os

import numpy as np

from zipline.utils.cli import maybe_show_progress

handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.handlers.append(handler)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parquet_equities(parquet_dir=None, tframe=None):
    """
    Generate an ingest function for custom data bundle
    This function can be used in ~/.zipline/extension.py
    to register bundle with custom parameters, e.g. with
    a custom trading calendar.

    Parameters
    ----------
    parquet_dir : string
        Structure of parquet file:
        store by field (symbol)
        date column name 'date'
    tframe: 'minute' or 'daily'
    Returns
    -------
    ingest : callable
        The bundle ingest function
    """

    return ParquetBundle(parquet_dir, tframe).ingest


class ParquetBundle:

    def __init__(self, parquet_dir=None, tframe=None):
        self.parquet_dir = parquet_dir
        self.tframe = tframe

    def ingest(
            self,
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
    ):
        parquet_bundle(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
            self.parquet_dir,
            self.tframe
        )


def parquet_bundle(
        environ,
        asset_db_writer,
        minute_bar_writer,
        daily_bar_writer,
        adjustment_writer,
        calendar,
        start_session,
        end_session,
        cache,
        show_progress,
        output_dir,
        parquet_dir=None,
        tframe=None
):
    if not parquet_dir:
        raise ValueError("parquet file is not given")

    divs_splits = {
        "divs": pd.DataFrame(
            columns=[
                "sid",
                "amount",
                "ex_date",
                "record_date",
                "declared_date",
                "pay_date",
            ]
        ),
        "splits": pd.DataFrame(columns=["sid", "ratio", "effective_date"]),
    }

    symbols = [f.split('.')[0] for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not symbols:
        raise ValueError("no symbol found in %s" % parquet_dir)

    dtype = [
        ("start_date", "datetime64[ns]"),
        ("end_date", "datetime64[ns]"),
        ("auto_close_date", "datetime64[ns]"),
        ("symbol", "object"),
    ]
    metadata = pd.DataFrame(np.empty(len(symbols), dtype=dtype))

    if tframe == "minute":
        writer = minute_bar_writer
    else:
        writer = daily_bar_writer

    writer.write(
        _pricing_iter(parquet_dir, symbols, metadata, divs_splits, show_progress, start_session, end_session, calendar),
        show_progress=show_progress,
    )

    metadata['exchange'] = 'NYSE'
    # 检测元数据中 symbol 是否为空值，然后清理空值
    metadata.dropna(subset=["symbol"], inplace=True)

    # 如果仍有 symbol 列为空，提示处理异常
    if metadata["symbol"].isnull().any():
        raise ValueError("Detected null 'symbol' values after cleaning metadata.")

    exchange = {'exchange': 'NYSE', 'canonical_name': 'NYSE', 'country_code': 'US'}
    exchange_df = pd.DataFrame(exchange, index=[0])
    asset_db_writer.write(equities=metadata, exchanges=exchange_df)

    divs_splits["divs"]["sid"] = divs_splits["divs"]["sid"].astype(int)
    divs_splits["splits"]["sid"] = divs_splits["splits"]["sid"].astype(int)
    adjustment_writer.write(
        splits=divs_splits["splits"], dividends=divs_splits["divs"]
    )


def _pricing_iter(parquet_dir, symbols, metadata, divs_splits, show_progress, start_session, end_session, calendar):
    with maybe_show_progress(
            symbols, show_progress, label="Loading pricing data: "
    ) as it:
        for sid, symbol in enumerate(it):
            logger.debug(f"{symbol}: sid {sid}")
            dfr = pd.read_parquet(os.path.join(parquet_dir, f"{symbol}.parquet"))
            if dfr.empty or dfr is None:
                continue
            if 'date' not in dfr.columns:
                continue
            dfr['date'] = pd.to_datetime(dfr['date'])
            dfr.set_index('date', inplace=True)
            dfr = dfr.sort_index()
            dfr = dfr.loc[start_session:end_session]
            if dfr.empty or dfr is None:
                continue

            # 从交易日历中获取交易日期范围
            all_sessions_in_calendar = calendar.sessions_in_range(start_session, end_session)

            # 对 DataFrame 重新索引，以确保所有交易日都在
            dfr = dfr.reindex(all_sessions_in_calendar, method=None)
            missing_ratio = dfr.isnull().mean().mean()
            if missing_ratio > 0:
                continue

            # if missing_ratio > 0:
            #     dfr.ffill(inplace=True)
            #     dfr.bfill(inplace=True)

            start_date = dfr.index[0]
            end_date = dfr.index[-1]

            # The auto_close date is the day after the last trade.
            ac_date = end_date + pd.Timedelta(days=1)
            metadata.iloc[sid] = start_date, end_date, ac_date, symbol

            if "split" in dfr.columns:
                tmp = 1.0 / dfr[dfr["split"] != 1.0]["split"]
                split = pd.DataFrame(
                    data=tmp.index.tolist(), columns=["effective_date"]
                )
                split["ratio"] = tmp.tolist()
                split["sid"] = sid

                splits = divs_splits["splits"]
                index = pd.Index(
                    range(splits.shape[0], splits.shape[0] + split.shape[0])
                )
                split.set_index(index, inplace=True)
                if not split.empty:
                    divs_splits["splits"] = pd.concat([divs_splits["splits"], split], axis=0)

            if "dividend" in dfr.columns:
                # ex_date   amount  sid record_date declared_date pay_date
                tmp = dfr[dfr["dividend"] != 0.0]["dividend"]
                div = pd.DataFrame(data=tmp.index.tolist(), columns=["ex_date"])
                div["record_date"] = pd.NaT
                div["declared_date"] = pd.NaT
                div["pay_date"] = pd.NaT
                div["amount"] = tmp.tolist()
                div["sid"] = sid

                divs = divs_splits["divs"]
                ind = pd.Index(range(divs.shape[0], divs.shape[0] + div.shape[0]))
                div.set_index(ind, inplace=True)
                if not div.empty:
                    divs_splits["divs"] = pd.concat([divs_splits["divs"], div], axis=0)

            yield sid, dfr


def store_dataframe_to_parquet_by_symbol(df: pd.DataFrame, parquet_dir: str):
    """
    Stores a dataframe into separate Parquet files by 'symbol' field with support for appending chunks.

    Parameters
    ----------
    df : pd.DataFrame
        The data to store, should contain at least:
        ['symbol', 'date', 'high', 'low', 'close', 'open', 'volume', 'dividend', 'split'].
    parquet_fp : str
        Path to save Parquet files. Files will be stored in a directory
        where each file corresponds to a symbol.

    Raises
    ------
    ValueError
        If 'symbol' field or required columns are missing.
    """

    required_columns = {'symbol', 'date', 'high', 'low', 'close', 'open', 'volume'}
    # Ensure the dataframe contains all required columns
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain the following columns: {', '.join(required_columns)}"
        )

    # Ensure the parent directory exists
    os.makedirs(parquet_dir, exist_ok=True)

    # Group by 'symbol' and append each group to the corresponding parquet file
    for symbol, group in df.groupby('symbol'):
        symbol_file_path = os.path.join(parquet_dir, f"{symbol}.parquet")

        # Append the group to the Parquet file
        append_to_parquet(group, symbol_file_path)


def append_to_parquet(df: pd.DataFrame, file_path: str):
    """
    Appends a DataFrame to an existing Parquet file or creates a new one.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be written to the Parquet file.
    file_path : str
        Path of the Parquet file.
    """
    table = pa.Table.from_pandas(df)

    # If the file exists, append to it
    if os.path.exists(file_path):
        existing_table = pq.read_table(file_path)  # Read the existing data
        combined_table = pa.concat_tables([existing_table, table])  # Combine new and existing data
        pq.write_table(combined_table, file_path)  # Overwrite the file with combined data
    else:
        # Write a new Parquet file if it doesn't exist
        pq.write_table(table, file_path)
