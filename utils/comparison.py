# utils/comparison.py
import time
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def get_model_factory(model_type: str):
    if model_type == "ridge":
        return lambda: Ridge(alpha=1.0)
    elif model_type == "mlp":
        return lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    else:  # linear
        return lambda: LinearRegression()

def run_comparison_method(method_name: str, X, y, k: int, model_factory, cv: int, seed: int = 42):
    t0 = time.perf_counter()
    try:
        if method_name == "SelectKBest":
            skb = SelectKBest(f_regression, k=min(k, X.shape[1]))
            skb.fit(X.fillna(0), y)
            sel = list(X.columns[skb.get_support()])
        elif method_name == "LassoCV":
            lasso = LassoCV(cv=5, random_state=seed, max_iter=5000).fit(X.fillna(0), y)
            sel = [c for c, coef in zip(X.columns, lasso.coef_) if abs(coef) > 1e-6]
        elif method_name == "RFE":
            rfe = RFE(LinearRegression(), n_features_to_select=min(k, X.shape[1]))
            rfe.fit(X.fillna(0), y)
            sel = list(X.columns[rfe.get_support()])
        elif method_name == "VarianceThreshold":
            var = X.var()
            sel = list(var.nlargest(k).index)
        elif method_name == "MutualInfo_topK":
            mi = mutual_info_regression(X.fillna(0), y, random_state=seed)
            mi_rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            sel = list(mi_rank.index[:k])
        elif method_name == "RandomForest_topK":
            rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1).fit(X.fillna(0), y)
            imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            sel = list(imp.index[:k])
        else:
            sel = []
        mse = None
        if sel:
            scores = cross_val_score(model_factory(), X[sel], y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            mse = -float(scores.mean())
        t1 = time.perf_counter()
        return {"selected": sel, "mse": mse, "time": t1 - t0}
    except Exception as e:
        t1 = time.perf_counter()
        return {"selected": [], "mse": None, "time": t1 - t0, "error": str(e)}