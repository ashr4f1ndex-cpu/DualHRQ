
import pandas as pd


def build_features_and_labels(S: pd.Series, iv_now: pd.Series, horizon: int=5, intraday_features: pd.DataFrame=None):
    """Daily features: returns, realized vol windows, IV level/slope + optional intraday aggregates.
    Label: future RV - current IV (proxy for IV-RV edge).
    """
    df = pd.DataFrame({"S": S, "iv": iv_now}).dropna()
    df["ret1"] = df["S"].pct_change()
    df["rv5"] = df["ret1"].rolling(5).std() * (252**0.5)
    df["rv20"] = df["ret1"].rolling(20).std() * (252**0.5)
    df["iv_slope"] = df["iv"].diff()
    # Merge intraday features if provided
    if intraday_features is not None and len(intraday_features) > 0:
        intraday_features = intraday_features.copy()
        intraday_features.index = pd.to_datetime(intraday_features.index)
        df = df.merge(intraday_features, left_index=True, right_index=True, how="left")
        # Fill forward limited to 1 day to avoid leakage
        df = df.fillna(method="ffill", limit=1)
    # Label: future realized vol minus current IV
    fut_rv = df["ret1"].rolling(horizon).std().shift(-horizon) * (252**0.5)
    df["label"] = fut_rv - df["iv"]
    df = df.dropna()
    features_cols = [c for c in df.columns if c not in ("label", "S")]
    features = df[features_cols]
    labels = df["label"]
    return features, labels
