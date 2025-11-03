"""Comparison methods for feature selection."""
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold, mutual_info_regression, mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score

def get_model_factory(model_type="linear", is_classification=False):
    """Create a model factory function based on type."""
    if is_classification:
        if model_type == "rf":
            def factory(): return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            def factory(): return LogisticRegression(random_state=42)
    else:
        if model_type == "rf":
            def factory(): return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            def factory(): return LinearRegression()
    return factory

def run_comparison_method(method_name, X, y, k, model_factory, cv=3, is_classification=False):
    """Run a comparison feature selection method."""
    t0 = time.perf_counter()
    try:
        if method_name == "SelectKBest":
            selector = SelectKBest(score_func=f_classif if is_classification else f_regression, k=k)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        
        elif method_name == "LassoCV":
            selector = LassoCV(cv=cv, random_state=42)
            selector.fit(X, y)
            selected = X.columns[abs(selector.coef_) > 1e-5].tolist()
            if not selected:  # If no features selected, take top k by coefficient magnitude
                selected = X.columns[np.argsort(abs(selector.coef_))[-k:]].tolist()
        
        elif method_name == "RFE":
            estimator = RandomForestClassifier(n_estimators=100) if is_classification else RandomForestRegressor(n_estimators=100)
            selector = RFE(estimator=estimator, n_features_to_select=k)
            selector.fit(X, y)
            selected = X.columns[selector.support_].tolist()
        
        elif method_name == "VarianceThreshold":
            # Use percentile-based threshold
            variances = X.var()
            threshold = np.percentile(variances, (100 * (len(X.columns) - k)) / len(X.columns))
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            selected = X.columns[selector.get_support()].tolist()
        
        elif method_name == "MutualInfo_topK":
            if is_classification:
                mi_scores = mutual_info_classif(X, y)
            else:
                mi_scores = mutual_info_regression(X, y)
            selected = X.columns[np.argsort(mi_scores)[-k:]].tolist()
        
        elif method_name == "RandomForest_topK":
            if is_classification:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            selected = X.columns[np.argsort(rf.feature_importances_)[-k:]].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Evaluate selected features
        X_selected = X[selected]
        model = model_factory()
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
        mse = cv_scores.mean() if is_classification else -cv_scores.mean()
        
        return {
            "selected": selected,
            "mse": mse,
            "time": time.perf_counter() - t0
        }
    
    except Exception as e:
        raise RuntimeError(f"Method {method_name} failed: {str(e)}")