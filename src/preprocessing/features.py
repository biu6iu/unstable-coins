from __future__ import annotations
import pandas as pd

class FeatureEngineer:
    """Adds technical-analysis features to the dataframe."""

    def sma(self, df: pd.DataFrame, window: int, column: str = "close") -> pd.DataFrame:
        """Add a simple moving average"""
        out = df.copy()
        out[f"sma_{window}"] = out[column].rolling(window=window).mean()
        return out

    def ema(self, df: pd.DataFrame, window: int, column: str = "close") -> pd.DataFrame:
        """Add an exponential moving average"""
        out = df.copy()
        out[f"ema_{window}"] = out[column].ewm(span=window, adjust=False).mean()
        return out

    def rsi(self, df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.DataFrame:
        """Add the Relative Strength Index using Wilder's smoothing (an EWM with alpha=1/window)"""
        out = df.copy()
        delta = out[column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha= 1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha= 1 / window, adjust=False, min_periods=window).mean()

        rs = avg_gain / avg_loss
        out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
        return out

    def returns(self, df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """Add percentage returns"""
        out = df.copy()
        out["returns"] = out[column].pct_change()
        return out

    def volatility(self, df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.DataFrame:
        """Add rolling volatility (std dev of returns)"""
        out = df.copy()
        rets = out[column].pct_change()
        out[f"volatility_{window}"] = rets.rolling(window=window).std()
        return out
