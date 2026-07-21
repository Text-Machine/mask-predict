"""
regression_analysis.py

A regression toolkit for a bounded dependent variable y and a set of
independent variables x on different numeric scales.

Two supported ranges for y:
  - [0, 1]   e.g. a rate, share, or probability
  - [-1, 1]  e.g. a similarity score, correlation-like metric, or
             signed attribution score

Because a bounded y is not necessarily binary, ordinary OLS can predict
values outside the valid range and mis-estimates variance near the
boundaries. The default model is therefore a "fractional logit"
(GLM with a Binomial family / logit link) fit on y rescaled to [0, 1]
internally -- the standard approach for bounded responses (Papke &
Wooldridge, 1996). All results are reported back on the original scale.
OLS is also available for comparison.

Main entry point: the RegressionAnalysis class.

Example (y in [0, 1])
----------------------
    ra = RegressionAnalysis(df, y="conversion_rate",
                             x=["price", "discount", "traffic"],
                             y_range="0_1")
    ra.fit(model="fractional_logit")
    print(ra.summary())

Example (y in [-1, 1])
-----------------------
    ra = RegressionAnalysis(df, y="attribution_score",
                             x=["freq", "position", "length"],
                             y_range="-1_1")
    ra.fit(model="fractional_logit")
    ra.full_report()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns


_VALID_RANGES = {
    "0_1": (0.0, 1.0),
    "-1_1": (-1.0, 1.0),
}


class RegressionAnalysis:
    """
    Fit, inspect, and visualise a regression of a bounded dependent
    variable on a set of independent variables.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    y : str
        Column name of the dependent variable.
    x : list[str]
        Column names of the independent variables.
    y_range : str, default "0_1"
        The valid range of y:
        - "0_1"  : y in [0, 1]   (rates, shares, probabilities)
        - "-1_1" : y in [-1, 1]  (signed scores, e.g. similarity,
                    correlation-like or signed attribution measures)
    scaler : str, default "standard"
        How to normalize x before fitting:
        - "standard": zero mean, unit variance (z-scores). Recommended
          default; keeps coefficients comparable across variables with
          different units/ranges.
        - "minmax": rescale each x to [0, 1].
        - None: use raw x values (not recommended when ranges differ).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        x: list[str],
        y_range: str = "0_1",
        scaler: str | None = "standard",
    ):
        if y_range not in _VALID_RANGES:
            raise ValueError(f"y_range must be one of {list(_VALID_RANGES)}")

        self.df_raw = df.copy()
        self.y_name = y
        self.x_names = list(x)
        self.y_range = y_range
        self.y_lo, self.y_hi = _VALID_RANGES[y_range]
        self.scaler_type = scaler

        self._validate_inputs()
        self._prepare_data()

        self.model = None
        self.results = None
        self.model_type = None

    # ------------------------------------------------------------------
    # Setup / validation
    # ------------------------------------------------------------------
    def _validate_inputs(self):
        missing = [c for c in [self.y_name, *self.x_names] if c not in self.df_raw.columns]
        if missing:
            raise ValueError(f"Column(s) not found in dataframe: {missing}")

        y_vals = self.df_raw[self.y_name].dropna()
        if y_vals.min() < self.y_lo or y_vals.max() > self.y_hi:
            raise ValueError(
                f"y ('{self.y_name}') must be within [{self.y_lo}, {self.y_hi}] "
                f"for y_range='{self.y_range}'. "
                f"Observed range: [{y_vals.min()}, {y_vals.max()}]"
            )

        non_numeric = [c for c in self.x_names if not pd.api.types.is_numeric_dtype(self.df_raw[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric independent variable(s): {non_numeric}")

    def _prepare_data(self):
        # Drop rows with missing values in the columns we need
        cols = [self.y_name, *self.x_names]
        self.data = self.df_raw[cols].dropna().reset_index(drop=True)

        self.y = self.data[self.y_name].astype(float)

        # Rescale y to [0, 1] internally for the fractional-logit model.
        # For y_range="0_1" this is a no-op; for "-1_1" it maps
        # -1 -> 0, 0 -> 0.5, 1 -> 1.
        self.y_unit = (self.y - self.y_lo) / (self.y_hi - self.y_lo)

        # Guard against exact 0/1 values, which break the logit link
        eps = 1e-4
        self.y_model = self.y_unit.clip(lower=eps, upper=1 - eps)

        x_raw = self.data[self.x_names].astype(float)

        if self.scaler_type == "standard":
            self._scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self._scaler = MinMaxScaler()
        elif self.scaler_type is None:
            self._scaler = None
        else:
            raise ValueError("scaler must be 'standard', 'minmax', or None")

        if self._scaler is not None:
            x_scaled = self._scaler.fit_transform(x_raw)
            self.x = pd.DataFrame(x_scaled, columns=self.x_names, index=x_raw.index)
        else:
            self.x = x_raw

    def _to_original_scale(self, y_unit: np.ndarray) -> np.ndarray:
        """Map a value (or array) from internal [0,1] scale back to y_range."""
        return self.y_lo + y_unit * (self.y_hi - self.y_lo)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, model: str = "fractional_logit"):
        """
        Fit the regression.

        model : {"fractional_logit", "ols"}
            - "fractional_logit" (default, recommended): GLM Binomial
              family with logit link, fit on y rescaled to [0, 1].
              Predictions are always mapped back into y_range and can
              never fall outside it.
            - "ols": plain linear regression on y in its original
              scale, for comparison only (can predict outside y_range).
        """
        X = sm.add_constant(self.x)

        if model == "fractional_logit":
            self.model = sm.GLM(self.y_model, X, family=sm.families.Binomial())
            self.results = self.model.fit()
        elif model == "ols":
            self.model = sm.OLS(self.y, X)
            self.results = self.model.fit()
        else:
            raise ValueError("model must be 'fractional_logit' or 'ols'")

        self.model_type = model
        self._X = X
        return self.results

    # ------------------------------------------------------------------
    # Summary / post-analysis
    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        return self.results.summary()

    def predictions(self) -> pd.DataFrame:
        """Return a dataframe of actual vs. fitted (predicted) y, on the original y_range scale."""
        self._check_fitted()
        if self.model_type == "ols":
            fitted = np.asarray(self.results.fittedvalues)
        else:
            fitted_unit = np.asarray(self.results.predict(self._X))
            fitted = self._to_original_scale(fitted_unit)
        return pd.DataFrame({"actual": self.y.values, "predicted": fitted})

    def residuals(self) -> pd.Series:
        self._check_fitted()
        preds = self.predictions()
        return preds["actual"] - preds["predicted"]

    def performance_metrics(self) -> dict:
        """Common fit metrics: R2 (pseudo for GLM), RMSE, MAE."""
        self._check_fitted()
        preds = self.predictions()
        resid = preds["actual"] - preds["predicted"]
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))

        if self.model_type == "ols":
            r2 = self.results.rsquared
            adj_r2 = self.results.rsquared_adj
        else:
            # McFadden's pseudo R^2 for GLM Binomial (computed on internal [0,1] scale)
            ll_full = self.results.llf
            null_model = sm.GLM(self.y_model, np.ones(len(self.y_model)),
                                 family=sm.families.Binomial()).fit()
            ll_null = null_model.llf
            r2 = 1 - (ll_full / ll_null)
            adj_r2 = np.nan  # not standard for pseudo-R2

        return {
            "model": self.model_type,
            "y_range": self.y_range,
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "mae": mae,
            "n_obs": int(self.results.nobs),
            "aic": self.results.aic,
        }

    def coefficients(self) -> pd.DataFrame:
        """
        Coefficient table with confidence intervals and significance.

        Note: for "fractional_logit", coefficients are on the logit
        scale of the internally rescaled [0,1] y, not the original
        y_range -- their sign and relative magnitude are directly
        interpretable, but the scale itself is not in y's native units.
        """
        self._check_fitted()
        conf_int = self.results.conf_int()
        conf_int.columns = ["ci_lower", "ci_upper"]
        table = pd.DataFrame({
            "coef": self.results.params,
            "std_err": self.results.bse,
            "p_value": self.results.pvalues,
        }).join(conf_int)
        table["significant_at_0.05"] = table["p_value"] < 0.05
        return table.round(4)

    def vif(self) -> pd.DataFrame:
        """
        Variance Inflation Factor per independent variable, to flag
        multicollinearity (VIF > 5-10 is typically concerning).
        """
        X = sm.add_constant(self.x)
        vif_data = pd.DataFrame({
            "variable": X.columns,
            "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        })
        return vif_data[vif_data["variable"] != "const"].reset_index(drop=True)

    def _check_fitted(self):
        if self.results is None:
            raise RuntimeError("Call .fit() before requesting results.")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_actual_vs_predicted(self, ax=None):
        self._check_fitted()
        preds = self.predictions()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(preds["predicted"], preds["actual"], alpha=0.5, edgecolor="k", linewidth=0.3)
        lims = [self.y_lo, self.y_hi]
        ax.plot(lims, lims, "r--", linewidth=1, label="perfect fit")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Actual vs. Predicted ({self.model_type}, y in {lims})")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()
        plt.tight_layout()
        return ax

    def plot_residuals(self, ax=None):
        self._check_fitted()
        preds = self.predictions()
        resid = preds["actual"] - preds["predicted"]
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(preds["predicted"], resid, alpha=0.5, edgecolor="k", linewidth=0.3)
        ax[0].axhline(0, color="r", linestyle="--", linewidth=1)
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Residual (actual - predicted)")
        ax[0].set_title("Residuals vs. Predicted")

        sns.histplot(resid, kde=True, ax=ax[1])
        ax[1].set_title("Residual Distribution")
        ax[1].set_xlabel("Residual")
        plt.tight_layout()
        return ax

    def plot_coefficients(self, ax=None):
        """Coefficient plot with 95% confidence intervals (excludes intercept)."""
        self._check_fitted()
        table = self.coefficients().drop(index="const", errors="ignore")
        table = table.sort_values("coef")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(table))))

        y_pos = np.arange(len(table))
        errors = [
            table["coef"] - table["ci_lower"],
            table["ci_upper"] - table["coef"],
        ]
        ax.errorbar(table["coef"], y_pos, xerr=errors, fmt="o", color="steelblue", capsize=3)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(table.index)
        scale_note = "logit scale, internally rescaled y" if self.model_type == "fractional_logit" else "raw y scale"
        ax.set_xlabel(f"Coefficient (95% CI) [{scale_note}]")
        title_suffix = "standardized x" if self.scaler_type == "standard" else "x as prepared"
        ax.set_title(f"Coefficient Estimates ({self.model_type}, {title_suffix})")
        plt.tight_layout()
        return ax

    def plot_correlation_heatmap(self, ax=None):
        """Correlation among independent variables (pre-scaling values)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.8 * len(self.x_names) + 2, 0.8 * len(self.x_names) + 2))
        corr = self.data[self.x_names].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Independent Variable Correlation Matrix")
        plt.tight_layout()
        return ax

    def plot_y_distribution(self, ax=None):
        """Histogram of y, useful to sanity-check the assumed y_range."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.y, bins=30, kde=True, ax=ax)
        ax.set_xlim(self.y_lo, self.y_hi)
        ax.set_title(f"Distribution of {self.y_name} (range: [{self.y_lo}, {self.y_hi}])")
        ax.set_xlabel(self.y_name)
        plt.tight_layout()
        return ax

    def full_report(self):
        """Print summary + metrics + coefficients + VIF, and show all plots."""
        self._check_fitted()
        print(self.summary())
        print("\nPerformance metrics:")
        for k, v in self.performance_metrics().items():
            print(f"  {k}: {v}")
        print("\nCoefficients:")
        print(self.coefficients())
        print("\nVariance Inflation Factors:")
        print(self.vif())

        self.plot_y_distribution()
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_coefficients()
        self.plot_correlation_heatmap()
        plt.show()


if __name__ == "__main__":
    # --- Minimal runnable examples with synthetic data ---
    rng = np.random.default_rng(0)
    n = 500

    # Example 1: y in [0, 1] (e.g. conversion rate)
    demo1 = pd.DataFrame({
        "price": rng.uniform(10, 500, n),
        "discount_pct": rng.uniform(0, 0.5, n),
        "ad_spend": rng.uniform(0, 10000, n),
    })
    logit1 = (
        -1.0
        - 0.01 * (demo1["price"] - demo1["price"].mean())
        + 4.0 * demo1["discount_pct"]
        + 0.0002 * demo1["ad_spend"]
        + rng.normal(0, 0.5, n)
    )
    demo1["conversion_rate"] = 1 / (1 + np.exp(-logit1))

    print("=== Example 1: y in [0, 1] ===")
    ra1 = RegressionAnalysis(demo1, y="conversion_rate", x=["price", "discount_pct", "ad_spend"], y_range="0_1")
    ra1.fit(model="fractional_logit")
    print(ra1.performance_metrics())

    # Example 2: y in [-1, 1] (e.g. a signed attribution / similarity score)
    demo2 = pd.DataFrame({
        "word_freq": rng.uniform(0, 1000, n),
        "sentence_position": rng.uniform(0, 1, n),
        "token_length": rng.uniform(1, 20, n),
    })
    signal = (
        0.4 * (demo2["sentence_position"] - 0.5)
        - 0.0006 * demo2["word_freq"]
        + 0.02 * demo2["token_length"]
        + rng.normal(0, 0.3, n)
    )
    demo2["attribution_score"] = np.tanh(signal)  # bounded to (-1, 1)

    print("\n=== Example 2: y in [-1, 1] ===")
    ra2 = RegressionAnalysis(demo2, y="attribution_score",
                              x=["word_freq", "sentence_position", "token_length"],
                              y_range="-1_1")
    ra2.fit(model="fractional_logit")
    print(ra2.performance_metrics())
    print(ra2.coefficients())
