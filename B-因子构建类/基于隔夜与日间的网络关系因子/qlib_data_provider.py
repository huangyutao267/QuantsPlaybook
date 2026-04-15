import logging
import os
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd

try:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    qlib = None
    D = None
    REG_CN = "REG_CN"
    QLIB_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QlibConfig:
    database_uri: str = os.getenv(
        "DOLPHINDB_URI",
        "dolphindb://username:password@host:port",
    )
    region: str = REG_CN


qlib_config = QlibConfig()


class QlibDataProvider:
    _qlib_initialized: bool = False

    DAILY_RETURN_EXPR: str = "$close/$preclose-1"
    DAYTIME_EXPR: str = "$close/$open-1"
    OVERNIGHT_EXPR: str = f"(1+{DAILY_RETURN_EXPR})/(1+{DAYTIME_EXPR})-1"

    @classmethod
    def init_qlib_once(cls) -> None:
        if not QLIB_AVAILABLE:
            return
        if cls._qlib_initialized:
            return
        qlib.init(
            database_uri=qlib_config.database_uri,
            region=qlib_config.region,
        )
        cls._qlib_initialized = True
        logger.info("Initialized qlib.")

    def __init__(self, codes: Union[List[str], str], start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self._use_synthetic = (not QLIB_AVAILABLE) or ("username:password@host:port" in qlib_config.database_uri)
        if not self._use_synthetic:
            self.init_qlib_once()
        self.instruments_ = self._parse_instruments(codes)
        self._pivot_features = (
            self._create_synthetic_features(start_date, end_date)
            if self._use_synthetic
            else self._fetch_and_pivot_features(start_date, end_date)
        )

    def _parse_instruments(self, codes: Union[List[str], str]) -> List[str]:
        if isinstance(codes, str):
            if codes not in ["csi300", "csi500", "ashares"]:
                raise ValueError(f"Unsupported codes value: {codes}")
            if self._use_synthetic:
                synthetic_sizes = {"csi300": 30, "csi500": 50, "ashares": 80}
                n_stocks = synthetic_sizes.get(codes, 30)
                return [f"STK{i:04d}" for i in range(1, n_stocks + 1)]
            return D.instruments(market=codes)
        if isinstance(codes, list):
            return codes
        raise TypeError("`codes` must be a list[str] or str")

    def _fetch_and_pivot_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        features: pd.DataFrame = D.features(
            self.instruments_,
            [self.DAILY_RETURN_EXPR, self.DAYTIME_EXPR, self.OVERNIGHT_EXPR],
            start_time=start_date,
            end_time=end_date,
        )
        features.rename(
            columns={
                self.DAILY_RETURN_EXPR: "daily_return",
                self.DAYTIME_EXPR: "daytime_return",
                self.OVERNIGHT_EXPR: "overnight_return",
            },
            inplace=True,
        )
        return pd.pivot_table(
            features,
            index="datetime",
            columns="instrument",
            values=["daily_return", "daytime_return", "overnight_return"],
        )

    def _create_synthetic_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq=pd.tseries.offsets.BDay())
        n_days = len(dates)
        n_stocks = len(self.instruments_)
        rng = np.random.default_rng(42)

        market = rng.normal(0.0002, 0.008, n_days)
        style = rng.normal(0.0, 0.004, (n_days, 3))
        loadings = rng.normal(0.0, 1.0, (n_stocks, 3))
        idio = rng.normal(0.0, 0.01, (n_days, n_stocks))

        daily = market[:, None] + style @ loadings.T * 0.2 + idio
        overnight = daily * 0.45 + rng.normal(0.0, 0.004, (n_days, n_stocks))
        daytime = ((1 + daily) / np.clip(1 + overnight, 1e-6, None)) - 1

        frames = {
            "daily_return": pd.DataFrame(daily, index=dates, columns=self.instruments_),
            "daytime_return": pd.DataFrame(daytime, index=dates, columns=self.instruments_),
            "overnight_return": pd.DataFrame(overnight, index=dates, columns=self.instruments_),
        }
        pivot = pd.concat(frames, axis=1)
        pivot.index.name = "datetime"
        pivot.columns.names = [None, "instrument"]
        logger.warning("Using synthetic market data fallback.")
        return pivot

    @property
    def daily_return_df(self) -> pd.DataFrame:
        return self._pivot_features["daily_return"]

    @property
    def daytime_return_df(self) -> pd.DataFrame:
        return self._pivot_features["daytime_return"]

    @property
    def overnight_return_df(self) -> pd.DataFrame:
        return self._pivot_features["overnight_return"]


def get_trade_days(end_date: str, count: int) -> List[pd.Timestamp]:
    all_days = pd.date_range(start="2000-01-01", end="2060-12-31", freq=pd.tseries.offsets.BDay())
    target_end = pd.Timestamp(end_date)
    if target_end not in all_days:
        target_end = all_days.asof(target_end)
    idx = all_days.get_loc(target_end)
    target_idx = idx + count
    if target_idx > len(all_days) - 1 or target_idx < 0:
        raise ValueError("Requested trade day offset is out of range")
    return [all_days[target_idx]]
