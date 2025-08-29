"""
Data preprocessing and encoding for quantum compatibility.

This module provides functions to preprocess and encode classical data
into a format suitable for quantum processing with 8 qubits.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Constants
RANDOM_STATE = 42
NUMERIC = ["OwnersAge", "VehicleAge", "BonusClass"]
CATEG = ["Gender", "Zone", "Class"]  # treat Zone & Class as categories even if int-coded


def load_clean(path: str) -> pd.DataFrame:
    """
    Load and clean the MCC dataset.
    
    Args:
        path: Path to the MCC dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = pd.read_csv(path)
    
    # Keep positive exposure
    df = df.loc[df["Duration"] > 0].copy()
    
    # Normalize gender labels (Swedish dataset uses 'M'/'K')
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M": "M", "K": "F"}).fillna(df["Gender"])
    
    # Ensure expected dtypes
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    for c in ["Zone", "Class"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    
    # Targets and helpers
    df["exposure"] = df["Duration"]
    df["claims"] = df["NumberClaims"].astype(int)
    df["total_cost"] = df["ClaimCost"].astype(float)
    df["occur"] = (df["claims"] > 0).astype(int)  # binary for frequency classifier
    
    # Severity only for claims > 0
    df["severity"] = np.where(df["claims"] > 0, df["total_cost"] / df["claims"], np.nan)
    
    return df


def make_quantum_feature_pipeline(k_qubits: int = 8) -> Pipeline:
    """
    Build a pipeline for quantum feature encoding:
      raw -> [OHE categoricals | scale numerics] -> PCA(k_qubits) -> angle map to [0, π]
    
    Args:
        k_qubits: Number of qubits to use for encoding
        
    Returns:
        Pipeline: Scikit-learn pipeline for quantum feature encoding
    """
    to_str = FunctionTransformer(lambda X: X.astype(str))
    cat = Pipeline([
        ("to_str", to_str),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    num = Pipeline([
        ("scale", StandardScaler()),
    ])
    
    pre = ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", cat, CATEG)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # PCA to the number of qubits you want to drive
    pca = PCA(n_components=k_qubits, random_state=RANDOM_STATE, whiten=False)
    
    # Angle mapping: map standardized PCA features to [0, π] for Ry/Rz encodings
    def to_angles(X: np.ndarray) -> np.ndarray:
        # robust scale then squash
        Xc = (X - np.median(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)
        # tanh squashes to [-1,1] → map to [0, π]
        return (np.tanh(Xc) + 1.0) * (np.pi / 2.0)
    
    angle_map = FunctionTransformer(to_angles, validate=False)
    
    return Pipeline([
        ("pre", pre),
        ("pca", pca),
        ("angles", angle_map),
    ])


def stratified_take(X, y, w, n):
    """
    Take up to n items, keeping class ratio.
    
    Args:
        X: Features
        y: Labels
        w: Weights
        n: Number of items to take
        
    Returns:
        Tuple: (X, y, w) with n items
    """
    n1 = int(min((y == 1).sum(), n // 2))
    n0 = int(min((y == 0).sum(), n - n1))
    idx1 = np.where(y == 1)[0][:n1]
    idx0 = np.where(y == 0)[0][:n0]
    sel = np.concatenate([idx1, idx0])
    return X.iloc[sel], y[sel], w[sel]


def quantum_ready_splits(
    df: pd.DataFrame,
    k_qubits: int = 8,
    n_train_occ: int = 600,
    n_test_occ: int = 200,
    n_train_sev: int = 500,
    n_test_sev: int = 200,
    random_state: int = RANDOM_STATE
) -> Dict:
    """
    Produce quantum-friendly matrices for:
      - Occurrence (binary): X_occ, y_occ, exposure weights
      - Severity (regression on claims>0): X_sev, y_sev
    
    Args:
        df: Cleaned dataset
        k_qubits: Number of qubits to use for encoding
        n_train_occ: Number of training samples for occurrence model
        n_test_occ: Number of test samples for occurrence model
        n_train_sev: Number of training samples for severity model
        n_test_sev: Number of test samples for severity model
        random_state: Random state for reproducibility
        
    Returns:
        Dict: Dictionary containing the quantum-ready data splits
    """
    pipe = make_quantum_feature_pipeline(k_qubits)
    
    # ---- Occurrence subset (keep class balance via stratified sampling)
    df_occ = df.copy()
    X_occ_raw = df_occ[NUMERIC + CATEG]
    y_occ = df_occ["occur"].to_numpy()
    w_occ = df_occ["exposure"].to_numpy()
    
    X_train_r, X_test_r, y_train, y_test, w_train, w_test = train_test_split(
        X_occ_raw, y_occ, w_occ, test_size=0.25, random_state=random_state, stratify=y_occ
    )
    
    # downsample for kernel tractability
    X_train_r = X_train_r.reset_index(drop=True)
    X_test_r = X_test_r.reset_index(drop=True)
    
    X_train_r, y_train, w_train = stratified_take(X_train_r, y_train, w_train, n_train_occ)
    X_test_r, y_test, w_test = stratified_take(X_test_r, y_test, w_test, n_test_occ)
    
    # fit encoder on train only
    X_train_occ = pipe.fit_transform(X_train_r)
    X_test_occ = pipe.transform(X_test_r)
    
    # ---- Severity subset (claims > 0)
    df_sev = df.loc[df["claims"] > 0].copy()
    X_sev_raw = df_sev[NUMERIC + CATEG]
    y_sev = df_sev["severity"].to_numpy()
    
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_sev_raw, y_sev, test_size=0.25, random_state=random_state
    )
    
    # uniform downsample
    Xs_tr = Xs_tr.reset_index(drop=True).iloc[:n_train_sev]
    ys_tr = ys_tr[:n_train_sev]
    Xs_te = Xs_te.reset_index(drop=True).iloc[:n_test_sev]
    ys_te = ys_te[:n_test_sev]
    
    X_train_sev = pipe.transform(Xs_tr)  # reuse same fitted pipe for consistency
    X_test_sev = pipe.transform(Xs_te)
    
    return {
        "pipe": pipe,
        "occ": (X_train_occ, y_train, w_train, X_test_occ, y_test, w_test),
        "sev": (X_train_sev, ys_tr, X_test_sev, ys_te),
    }


def analyze_encoded_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    name: str = "Dataset"
) -> Dict:
    """
    Analyze and report statistics about encoded data.
    
    Args:
        X_train: Encoded features
        y_train: Labels
        name: Name of the dataset
        
    Returns:
        Dict: Dictionary of statistics about the encoded data
    """
    stats = {
        "name": name,
        "shape": X_train.shape,
        "n_qubits": X_train.shape[1],
        "min": float(np.min(X_train)),
        "max": float(np.max(X_train)),
        "mean": float(np.mean(X_train)),
        "std": float(np.std(X_train))
    }
    
    if len(np.unique(y_train)) <= 5:  # Likely classification
        stats["class_distribution"] = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    else:  # Likely regression
        stats["target_min"] = float(np.min(y_train))
        stats["target_max"] = float(np.max(y_train))
        stats["target_mean"] = float(np.mean(y_train))
        stats["target_std"] = float(np.std(y_train))
    
    return stats