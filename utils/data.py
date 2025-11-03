import pandas as pd
from pathlib import Path

def prepare_data_for_example(path: str, group_top: int, drop_numeric_features: bool, target: str = None, drop_cols=None, max_cardinality: int = None):
    df = pd.read_csv(path)
    if drop_cols:
        to_drop = [c for c in drop_cols if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_proc = df.copy()
    if max_cardinality is not None:
        cat_card = {c: df_proc[c].nunique(dropna=True) for c in df_proc.columns if df_proc[c].dtype == 'object' or str(df_proc[c].dtype).startswith('category')}
        drop_high = [c for c, card in cat_card.items() if card > max_cardinality]
        if drop_high:
            df_proc = df_proc.drop(columns=drop_high)
    if group_top and group_top > 0:
        for c in cat_cols:
            top = df_proc[c].value_counts().nlargest(group_top).index
            df_proc[c] = df_proc[c].where(df_proc[c].isin(top), other='OTHER')
    if cat_cols:
        df_enc = pd.get_dummies(df_proc, columns=cat_cols, dummy_na=False)
    else:
        df_enc = df_proc.copy()
    numeric = df_enc.select_dtypes(include=['number'])
    numeric_cols = numeric.columns.tolist()
    if target is not None:
        if target not in df_enc.columns:
            raise RuntimeError(f"Target column '{target}' not found in CSV after encoding")
        target_col = target
    else:
        if not numeric_cols:
            raise RuntimeError('No numeric columns found to use as target')
        target_col = numeric_cols[-1]
    if drop_numeric_features:
        numeric_feature_cols = [c for c in numeric.columns if c != target_col]
        df_enc = df_enc.drop(columns=numeric_feature_cols)
    y = df_enc[target_col]
    X = df_enc.drop(columns=[target_col])
    
    # Handle missing values BEFORE dropping
    numeric_X = X.select_dtypes(include=['number'])
    X_filled = X.copy()
    for col in numeric_X.columns:
        if X_filled[col].isna().any():
            X_filled[col].fillna(X_filled[col].mean(), inplace=True)
    if y.isna().any():
        y.fillna(y.median(), inplace=True)
    
    # Now align X and y without dropping rows
    valid = X_filled.join(y).dropna()
    X_final = valid.drop(columns=[target_col])
    y_final = valid[target_col]
    return X_final, y_final