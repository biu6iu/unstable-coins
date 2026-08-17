from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats

DEFAULT_CONFIDENCE = 0.95

@dataclass
class Estimate:
    value: float
    se: float
    ci_low: float
    ci_high: float
    method: str

    @classmethod
    def from_se(cls, value: float, se: float, method: str, confidence: float = DEFAULT_CONFIDENCE) -> "Estimate":
        """build a normal-approximation interval"""
        z = float(stats.norm.ppf(0.5 + confidence / 2))
        return cls(value=value, se=se, ci_low=value - z * se, ci_high=value + z * se, method=method)

    def excludes(self, null_value: float = 0.0) -> bool:
        """checks if the CI contains the null value, and if not, then the estimate is statistically significant"""
        return null_value < self.ci_low or null_value > self.ci_high

    def z_score(self, null_value: float = 0.0) -> float:
        """standardised distance from a stated null, the t-statistic every reported estimate implies"""
        return (self.value - null_value) / self.se if self.se else float("nan")

    def p_value(self, null_value: float = 0.0) -> float:
        """two-sided normal-approximation p-value against a stated null"""
        z = self.z_score(null_value)
        return float("nan") if np.isnan(z) else float(2 * stats.norm.sf(abs(z)))

    def __str__(self) -> str:
        return f"{self.value:+.4f} +/- {self.se:.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}] ({self.method})"


@dataclass
class HypothesisTest:
    statistic: float
    p_value: float
    null: str
    conclusion: str

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __str__(self) -> str:
        return f"statistic={self.statistic:+.4f}, p={self.p_value:.4f} | H0: {self.null} | {self.conclusion}"