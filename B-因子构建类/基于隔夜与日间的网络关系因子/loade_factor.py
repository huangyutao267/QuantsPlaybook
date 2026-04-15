import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    import qlib
    from qlib.config import REG_CN
except ImportError:
    qlib = None
    REG_CN = "REG_CN"

from factor_pipeline import FactorPipeline
from qlib_data_provider import QLIB_AVAILABLE, qlib_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_abbr(name: str) -> str:
    return "".join(w[0] for w in name.split("_") if w).lower()


def load_factor(start_dt: str, end_dt: str, network_type: str, method: str) -> None:
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = FactorPipeline(
        codes=os.getenv("FACTOR_CODES", "ashares"),
        start_dt=start_dt,
        end_dt=end_dt,
        window=int(os.getenv("FACTOR_WINDOW", "60")),
        network_type=network_type,
        correlation_method=method,
        top_percentile=0.2,
        bottom_percentile=0.2,
        lead_percentile=0.5,
    )
    pipeline.run()

    long_factor_df: pd.DataFrame = pipeline.long_df
    short_factor_df: pd.DataFrame = pipeline.short_df

    long_factor_df.where(long_factor_df != 0).to_parquet(
        out_dir / f"{to_abbr(network_type)}_{method}_long.parquet"
    )
    short_factor_df.where(short_factor_df != 0).to_parquet(
        out_dir / f"{to_abbr(network_type)}_{method}_short.parquet"
    )


network_types: List[str] = [
    "daytime_lead_overnight",
    "overnight_lead_daytime",
    "preclose_lead_close",
]
correlation_method: List[str] = ["pearson", "spearman"]


if __name__ == "__main__":
    if QLIB_AVAILABLE and qlib is not None and "username:password@host:port" not in qlib_config.database_uri:
        qlib.init(database_uri=qlib_config.database_uri, region=REG_CN)
    else:
        logger.warning("qlib/database unavailable, running with synthetic fallback data.")

    start_dt = os.getenv("FACTOR_START_DT", "2020-01-01")
    end_dt = os.getenv("FACTOR_END_DT", "2025-10-31")

    for net_type in network_types:
        for method in correlation_method:
            logger.info("Start loading factor for %s with %s...", net_type, method)
            load_factor(
                start_dt=start_dt,
                end_dt=end_dt,
                network_type=net_type,
                method=method,
            )
            logger.info("Finished loading factor for %s with %s.", net_type, method)

    logger.info("All factors have been loaded and saved successfully.")
