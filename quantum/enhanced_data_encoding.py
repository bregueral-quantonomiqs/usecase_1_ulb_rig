"""
Enhanced data preprocessing and encoding for quantum compatibility.

This module provides improved functions to preprocess and encode classical data
into a format suitable for quantum processing with focus on severity prediction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Try to import advanced resampling libraries
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import smogn
    HAS_SMOGN = True
except ImportError:
    HAS_SMOGN = False

# Constants
RANDOM_STATE = 42
NUMERIC = ["OwnersAge", "VehicleAge", "BonusClass"]
CATEG = ["Gender", "Zone", "Class"]  # treat Zone & Class as categories even if int-coded


def load_clean(path: str) -> pd.DataFrame:
    """
    Load and clean the MCC dataset with enhanced preprocessing.
    
    Args:
        path: Path to the MCC dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset with engineered features
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
    
    # Add engineered features
    df = add_engineered_features(df)
    
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific engineered features for better prediction.
    
    Args:
        df: Original dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    # Create a copy to avoid modifying the original
    df_enhanced = df.copy()
    
    # Add interaction terms
    df_enhanced["age_class"] = df_enhanced["OwnersAge"] * df_enhanced["Class"]
    df_enhanced["age_zone"] = df_enhanced["OwnersAge"] * df_enhanced["Zone"]
    
    # Add exposure-related features (important for insurance)
    df_enhanced["log_exposure"] = np.log1p(df_enhanced["exposure"])
    df_enhanced["exposure_bonus"] = df_enhanced["log_exposure"] * df_enhanced["BonusClass"]
    
    # Add non-linear transformations
    df_enhanced["age_squared"] = df_enhanced["OwnersAge"] ** 2
    df_enhanced["log_vehicle_age"] = np.log1p(df_enhanced["VehicleAge"])
    
    # Add risk score - combine multiple factors
    df_enhanced["risk_score"] = (
        df_enhanced["OwnersAge"] / 100 +
        df_enhanced["VehicleAge"] / 5 +
        df_enhanced["Zone"] / df_enhanced["Zone"].max() +
        (5 - df_enhanced["BonusClass"]) / 5
    )
    
    return df_enhanced


def make_enhanced_quantum_feature_pipeline(
    k_qubits: int = 8,
    supervised: bool = True,
    target_variable: Optional[np.ndarray] = None,
    feature_importance: Optional[np.ndarray] = None
) -> Pipeline:
    """
    Build an enhanced pipeline for quantum feature encoding:
      raw -> [OHE categoricals | scale numerics] -> feature selection ->
      supervised dim reduction -> advanced angle mapping
    
    Args:
        k_qubits: Number of qubits to use for encoding
        supervised: Whether to use supervised dimensionality reduction
        target_variable: Target variable for supervised methods
        feature_importance: Optional importance weights for features
        
    Returns:
        Pipeline: Scikit-learn pipeline for quantum feature encoding
    """
    # Enhanced categorical handling (direct OHE to avoid nested Pipeline warnings)
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # Enhanced numerical handling with dynamic n_quantiles to avoid warnings
    class SafeQuantileTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, n_quantiles: int = 1000, output_distribution: str = 'normal', subsample: Optional[int] = None,
                     random_state: Optional[int] = RANDOM_STATE, copy: bool = True):
            self.n_quantiles = n_quantiles
            self.output_distribution = output_distribution
            self.subsample = subsample
            self.random_state = random_state
            self.copy = copy
            self._qt = None

        def fit(self, X, y=None):
            # Keep DataFrame inputs intact to preserve feature names and avoid warnings
            n_samples = int(getattr(X, 'shape', (1,))[0] or 1)
            n_q = int(min(max(10, n_samples), self.n_quantiles))
            self._qt = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution=self.output_distribution,
                subsample=self.subsample,
                random_state=self.random_state,
                copy=self.copy,
            )
            self._qt.fit(X)
            return self

        def transform(self, X):
            if self._qt is None:
                self.fit(X)
            return self._qt.transform(X)

    # Enhanced numerical handling (single transformer, avoid nested Pipeline warnings)
    num = SafeQuantileTransformer(n_quantiles=1000, output_distribution='normal')
    
    # Preprocess all features
    pre = ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", cat, CATEG),
    ], remainder="drop", verbose_feature_names_out=False)
    
    # Advanced angle mapping using rankdata for better distribution
    def adaptive_angle_mapping(X: np.ndarray) -> np.ndarray:
        # Handle various input types
        X = np.asarray(X)
        
        # Handle 1D arrays and other shapes
        if X.ndim == 0:
            # Scalar - convert to 2D array
            X = np.array([[X]])
        elif X.ndim == 1:
            # 1D array - convert to 2D column vector
            X = X.reshape(-1, 1)
        
        # First normalize using rankdata for better handling of outliers
        X_ranked = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_ranked[:, i] = rankdata(X[:, i]) / len(X)
        
        # Apply Beta distribution to control the shape of the distribution
        # Parameters 2,5 give more weight to smaller values, appropriate for skewed insurance data
        X_shaped = stats.beta.ppf(X_ranked, 2, 5)
        
        # Map to [0, π] for rotation gates
        result = X_shaped * np.pi
        
        # Return with original dimensionality
        if X.shape[0] == 1 and X.shape[1] == 1:
            return float(result[0, 0])  # Return scalar
        elif X.shape[1] == 1:
            return result.ravel()  # Return 1D if input was 1D
        return result
    
    angle_map = FunctionTransformer(adaptive_angle_mapping, validate=False)
    
    # Build different pipelines based on whether feature selection and supervised dim reduction are used
    if supervised and target_variable is not None:
        # Use Partial Least Squares for supervised dimensionality reduction
        
        # Create a proper custom transformer by inheriting from sklearn base classes
        class PLSTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, n_components, target):
                self.n_components = n_components
                self.target = target
                self.pls_model = None
            
            def fit(self, X, y=None):
                try:
                    self.pls_model = PLSRegression(n_components=self.n_components, scale=False)
                    self.pls_model.fit(X, self.target)
                    return self
                except Exception as e:
                    print(f"Error in PLS fit: {e}")
                    # Fall back to PCA if PLS fails
                    from sklearn.decomposition import PCA
                    print("Falling back to PCA")
                    self.pls_model = PCA(n_components=min(self.n_components, X.shape[1]), random_state=RANDOM_STATE)
                    self.pls_model.fit(X)
                    return self
            
            def transform(self, X):
                try:
                    Z = self.pls_model.transform(X)
                    # Handle possible tuple outputs (e.g., (X_scores, Y_scores))
                    if isinstance(Z, tuple):
                        Z = Z[0]
                    # Ensure 2D shape (n_samples, n_components)
                    Z = np.asarray(Z)
                    if Z.ndim == 1:
                        Z = Z.reshape(-1, 1)
                    return Z
                except Exception as e:
                    print(f"Error in transform: {e}")
                    # Emergency fallback - just return X with reduced dimensions
                    if X.shape[1] > self.n_components:
                        return X[:, :self.n_components]
                    return X
        
        dim_red = PLSTransformer(n_components=k_qubits, target=target_variable)
        
        # Build pipeline with supervised dim reduction only
        return Pipeline([
            ("pre", pre),
            ("dim_red", dim_red),
            ("angles", angle_map),
        ])
    else:
        # Use PCA if not supervised or no target provided
        dim_red = PCA(n_components=min(k_qubits, 8), random_state=RANDOM_STATE, whiten=True)
        
        # Build pipeline with PCA
        return Pipeline([
            ("pre", pre),
            ("dim_red", dim_red),
            ("angles", angle_map),
        ])


def augment_severity_data(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment severity data to improve model performance on rare extreme values.
    
    Args:
        X: Feature matrix
        y: Target values (severity)
        n_samples: Desired number of samples after augmentation
        
    Returns:
        Tuple: (X_augmented, y_augmented)
    """
    if not HAS_SMOGN:
        print("Warning: smogn not installed. Using original data.")
        return X, y
    
    # Convert to DataFrame for smogn
    combined = np.column_stack([X, y])
    column_names = [f"feature_{i}" for i in range(X.shape[1])] + ["target"]
    df = pd.DataFrame(combined, columns=column_names)
    
    # Apply SMOGN for regression oversampling with focus on rare extreme values
    df_aug = smogn.smoter(
        data=df,
        y="target",
        k=5,
        samp_method="extreme",
        rel_thres=0.8,  # Focus on high values
        rel_method="auto",
        rel_xtrm_type="high",  # High values are more important in severity
        rel_coef=0.8
    )
    
    # Limit to desired sample size
    if len(df_aug) > n_samples:
        df_aug = df_aug.sample(n_samples, random_state=RANDOM_STATE)
    
    # Return as numpy arrays
    X_aug = df_aug.iloc[:, :-1].values
    y_aug = df_aug.iloc[:, -1].values
    
    return X_aug, y_aug


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


def calculate_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray
) -> np.ndarray:
    """
    Calculate feature importance for severity prediction.
    
    Args:
        X: Feature dataframe
        y: Target values
        
    Returns:
        np.ndarray: Feature importance scores
    """
    # Handle categorical features explicitly
    X_numeric = X.copy()
    
    # Convert categorical columns to numeric or drop them
    for col in X_numeric.columns:
        if X_numeric[col].dtype == object or pd.api.types.is_categorical_dtype(X_numeric[col]):
            # For categorical columns with few unique values, convert to one-hot
            if X_numeric[col].nunique() <= 10:
                # Create dummy variables and drop the original column
                dummies = pd.get_dummies(X_numeric[col], prefix=col, drop_first=True)
                X_numeric = pd.concat([X_numeric, dummies], axis=1)
            
            # Drop the original categorical column
            X_numeric = X_numeric.drop(columns=[col])
    
    # Safety check - ensure all columns are numeric
    for col in X_numeric.columns:
        if not pd.api.types.is_numeric_dtype(X_numeric[col]):
            print(f"Dropping non-numeric column: {col}")
            X_numeric = X_numeric.drop(columns=[col])
    
    if X_numeric.empty:
        print("Warning: No numeric features left after processing")
        # Return equal importance for all original features
        equal_importance = np.ones(X.shape[1]) / X.shape[1]
        return equal_importance
    
    # Calculate mutual information for numeric features
    try:
        mi_scores = mutual_info_regression(X_numeric, y)
        
        # Normalize scores
        importance = mi_scores / np.sum(mi_scores) if np.sum(mi_scores) > 0 else np.ones_like(mi_scores)
        
        # Match the length of importance scores to the number of original features
        if len(importance) != X.shape[1]:
            print(f"Adjusting importance scores from {len(importance)} to {X.shape[1]}")
            # Simple solution: create equal weights if dimensions don't match
            importance = np.ones(X.shape[1]) / X.shape[1]
            
        return importance
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        # Fallback to equal importance
        return np.ones(X.shape[1]) / X.shape[1]


def enhanced_quantum_ready_splits(
    df: pd.DataFrame,
    k_qubits: int = 8,
    n_train_occ: int = 600,
    n_test_occ: int = 200,
    n_train_sev: int = 500,
    n_test_sev: int = 200,
    augment_severity: bool = True,
    random_state: int = RANDOM_STATE
) -> Dict:
    """
    Produce enhanced quantum-friendly matrices with focus on severity prediction.
    
    Args:
        df: Cleaned dataset
        k_qubits: Number of qubits to use for encoding
        n_train_occ: Number of training samples for occurrence model
        n_test_occ: Number of test samples for occurrence model
        n_train_sev: Number of training samples for severity model
        n_test_sev: Number of test samples for severity model
        augment_severity: Whether to augment severity data
        random_state: Random state for reproducibility
        
    Returns:
        Dict: Dictionary containing the enhanced quantum-ready data splits
    """
    # ---- Occurrence subset (keep class balance via stratified sampling)
    df_occ = df.copy()
    
    # For occurrence, use all engineered features
    engineered_features = [
        "age_class", "age_zone", "log_exposure", "exposure_bonus", 
        "age_squared", "log_vehicle_age", "risk_score"
    ]
    
    # Check which engineered features exist
    available_features = [f for f in engineered_features if f in df_occ.columns]
    
    # Base features plus available engineered features
    X_occ_raw = df_occ[NUMERIC + CATEG + available_features]
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
    
    # Standard pipeline for occurrence
    pipe_occ = make_enhanced_quantum_feature_pipeline(k_qubits)
    
    # fit encoder on train only
    X_train_occ = pipe_occ.fit_transform(X_train_r)
    X_test_occ = pipe_occ.transform(X_test_r)
    
    # ---- Severity subset (claims > 0)
    df_sev = df.loc[df["claims"] > 0].copy()
    X_sev_raw = df_sev[NUMERIC + CATEG + available_features]
    y_sev = df_sev["severity"].to_numpy()
    
    # Get only numeric features for importance calculation
    X_sev_numeric = X_sev_raw.select_dtypes(include=np.number)
    print(f"Using {X_sev_numeric.shape[1]} numeric features for importance calculation")
    
    # Calculate feature importance for severity prediction using only numeric features
    feature_importance = calculate_feature_importance(X_sev_numeric, y_sev)
    
    # Use 75/25 split ratio consistently
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_sev_raw, y_sev, test_size=0.25, random_state=random_state
    )
    
    # Uniform downsample
    Xs_tr = Xs_tr.reset_index(drop=True).iloc[:n_train_sev]
    ys_tr = ys_tr[:n_train_sev]
    Xs_te = Xs_te.reset_index(drop=True).iloc[:n_test_sev]
    ys_te = ys_te[:n_test_sev]
    
    # Data augmentation for severity - disable for now to debug core functionality
    if augment_severity and HAS_SMOGN and False:  # Temporarily disabled
        print("Augmenting severity data...")
        Xs_tr_values = Xs_tr.values
        Xs_tr_values, ys_tr = augment_severity_data(Xs_tr_values, ys_tr, n_samples=n_train_sev*2)
        # Convert back to DataFrame with original column names
        Xs_tr = pd.DataFrame(Xs_tr_values, columns=Xs_tr.columns)
    
    # Enhanced supervised pipeline for severity with target information
    pipe_sev = make_enhanced_quantum_feature_pipeline(
        k_qubits=k_qubits,
        supervised=True,
        target_variable=ys_tr,
        feature_importance=None  # Disable feature importance temporarily
    )
    
    X_train_sev = pipe_sev.fit_transform(Xs_tr)
    X_test_sev = pipe_sev.transform(Xs_te)
    
    return {
        "pipe_occ": pipe_occ,
        "pipe_sev": pipe_sev,
        "occ": (X_train_occ, y_train, w_train, X_test_occ, y_test, w_test),
        "sev": (X_train_sev, ys_tr, X_test_sev, ys_te),
        "feature_importance": feature_importance
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
    from scipy import stats as scipy_stats
    
    # Handle different input shapes
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    # Print debugging info
    print(f"X_train shape in analyze_encoded_data: {X_train.shape}")
    
    # Ensure X_train is 2D
    if X_train.ndim == 0:
        X_train = np.array([[X_train]])
    elif X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    result_stats = {
        "name": name,
        "shape": X_train.shape,
        "n_features": X_train.shape[1] if X_train.ndim > 1 else 1,
    }
    
    # Add basic statistics
    try:
        result_stats.update({
            "min": float(np.min(X_train)),
            "max": float(np.max(X_train)),
            "mean": float(np.mean(X_train)),
            "std": float(np.std(X_train)),
            "median": float(np.median(X_train)),
        })
        
        # Skew can sometimes fail on certain distributions
        try:
            result_stats["skew"] = float(scipy_stats.skew(X_train.flatten()))
        except:
            result_stats["skew"] = "N/A"
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        result_stats.update({
            "min": "N/A",
            "max": "N/A",
            "mean": "N/A",
            "std": "N/A",
            "median": "N/A",
            "skew": "N/A",
        })
    
    # Add target information
    try:
        unique_values = np.unique(y_train)
        if len(unique_values) <= 5:  # Likely classification
            result_stats["class_distribution"] = {
                str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
            }
        else:  # Likely regression
            result_stats.update({
                "target_min": float(np.min(y_train)),
                "target_max": float(np.max(y_train)),
                "target_mean": float(np.mean(y_train)),
                "target_median": float(np.median(y_train)),
                "target_std": float(np.std(y_train)),
            })
            
            # Skew can sometimes fail on certain distributions
            try:
                result_stats["target_skew"] = float(scipy_stats.skew(y_train))
            except:
                result_stats["target_skew"] = "N/A"
    except Exception as e:
        print(f"Error calculating target statistics: {e}")
        result_stats.update({
            "target_stats": "Error calculating target statistics"
        })
    
    return result_stats
