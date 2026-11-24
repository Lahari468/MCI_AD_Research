"""
Baseline Machine Learning Classifiers
Module: src/models/baselines.py
Implements L2 Regularized Logistic Regression with Youden J operating threshold selection.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score


def select_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select optimal classification threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1).
    """
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)


class LogisticRegressionBaseline:
    """
    L2 Logistic Regression with nested validation for C tuning and Youden J threshold selection.
    """
    def __init__(self, c_grid=None, random_state: int = 42):
        self.c_grid = c_grid or [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.random_state = random_state
        self.best_c = 1.0
        self.best_threshold = 0.5
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit with optional inner validation tuning."""
        if X_val is not None and y_val is not None:
            best_auc = -1.0
            for c in self.c_grid:
                clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=self.random_state)
                clf.fit(X_train, y_train)
                prob_val = clf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, prob_val)
                if auc > best_auc:
                    best_auc = auc
                    self.best_c = c

            # Fit on train with best C to select Youden threshold on val
            clf_inner = LogisticRegression(C=self.best_c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=self.random_state)
            clf_inner.fit(X_train, y_train)
            prob_inner_val = clf_inner.predict_proba(X_val)[:, 1]
            self.best_threshold = select_threshold_youden(y_val, prob_inner_val)

        self.model = LogisticRegression(C=self.best_c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = threshold if threshold is not None else self.best_threshold
        return (self.predict_proba(X) >= t).astype(int)
