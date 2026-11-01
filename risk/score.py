"""Flow-aware risk score and the density-only baseline. Both expose score(df) -> array in 0..1."""
import numpy as np
from sklearn.linear_model import LogisticRegression

D_MAX = 6.0   # people / m^2 used to scale density to 0..1


def scaled(df):
    """Scale the three signals to 0..1 (columns must be smoothed already)."""
    d = np.clip(df["density"].to_numpy() / D_MAX, 0, 1)
    h = np.clip(df["entropy"].to_numpy(), 0, 1)
    s = np.clip(df["slowdown"].to_numpy(), 0, 1)
    return np.column_stack([d, h, s])


class DensityBaseline:
    """Alert on crowd density alone."""
    name = "density-only"

    def score(self, df):
        return scaled(df)[:, 0]


class FlowAwareRisk:
    """R = sigmoid(b + w1*D + w2*H + w3*S), weights learned by logistic regression."""
    name = "flow-aware"

    def fit(self, df, y):
        self.model = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000)
        self.model.fit(scaled(df), y)
        self.weights = dict(zip(["density", "entropy", "slowdown"], self.model.coef_[0].round(2)))
        return self

    def score(self, df):
        return self.model.predict_proba(scaled(df))[:, 1]
