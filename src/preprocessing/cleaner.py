from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DataCleaner:
    """Normalises a raw OHLCV DataFrame into a clean, gap-free series"""

    def __init__(self, max_ffill: int = 3) -> None:
        # max_ffill caps how many consecutive missing bars we'll paper over with forward-fill
        self.max_ffill = max_ffill

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # remove duplicates
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count:
            logger.warning("Dropping %d duplicate timestamp(s)", duplicate_count)
            df = df[~df.index.duplicated(keep="first")]

        df = df.sort_index()
        df = self._fill_gaps(df)
        df = df.astype(float)
        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill small gaps in an inferred, regular time grid."""
        if len(df) < 3:
            return df

        median_step = df.index.to_series().diff().median()
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=median_step)
        full_index.name = df.index.name
    
        missing = full_index.difference(df.index)
        if len(missing) == 0:
            return df

        reindexed = df.reindex(full_index)
        filled = reindexed.ffill(limit=self.max_ffill)

        still_missing = filled.isna().any(axis=1).sum()
        logger.warning(
            "Forward-filled %d gap bar(s); %d bar(s) remain unfillable (gap too large)",
            len(missing),
            still_missing,
        )

        return filled.dropna()
