import numpy as np


def empirical_risk_squared(y_true, y_pred) -> float:
    """R̂_S(h) = (1/n) Σ (y_i - h(x_i))²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def logistic_risk(y_true, p_hat, eps: float = 1e-12) -> float:
    """R̂ logístico = -(1/n) Σ [y log p + (1-y) log(1-p)].

    Se recorta p_hat a [eps, 1-eps] para evitar log(0).
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_hat, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def misclassification(y_true, y_pred) -> tuple[int, float]:
    """Devuelve (n_errores, proporción_de_errores)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.size
    errs = int(np.sum(y_true != y_pred))
    return errs, errs / n