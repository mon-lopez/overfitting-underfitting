"""Ajuste de modelos reutilizables."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass
class LinearFit:
    """Resultado de una regresión lineal con intercepto."""
    intercept: float
    coefs: np.ndarray
    feature_names: list[str]
    model: LinearRegression

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X).reshape(-1, len(self.feature_names)))


def fit_linear(df, x_cols, y_col="y") -> LinearFit:
    """Ajusta y ~ x_cols con intercepto.

    Devuelve un objeto con intercepto, coeficientes, nombres y el modelo.
    """
    X = df[list(x_cols)].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    return LinearFit(
        intercept=float(model.intercept_),
        coefs=np.asarray(model.coef_, dtype=float).ravel(),
        feature_names=list(x_cols),
        model=model,
    )


def fit_logistic(df, x_cols, y_col="abandona_30d",
                 C=1e6, max_iter=5000, solver="lbfgs") -> LogisticRegression:
    """Ajusta una regresión logística con los hiperparámetros del taller."""
    X = df[list(x_cols)].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    clf = LogisticRegression(solver=solver, C=C, max_iter=max_iter)
    clf.fit(X, y)
    return clf