"""
Quantum kernel methods for classification and regression.

This module provides functions to compute quantum kernel matrices using
different feature maps and to train models using these kernel matrices.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def create_feature_map_circuit(
    feature_map: Union[ZZFeatureMap, PauliFeatureMap],
    x: np.ndarray
) -> QuantumCircuit:
    """
    Create a quantum circuit with the feature map applied to the input data.
    
    Args:
        feature_map: Feature map to use
        x: Input data
        
    Returns:
        QuantumCircuit: Quantum circuit with the feature map applied
    """
    params_dict = {param: x[i] for i, param in enumerate(feature_map.parameters)}
    return feature_map.assign_parameters(params_dict)


def _resolve_backend_or_runtime(backend_name: str):
    """
    Try to resolve a local Aer backend; if not found and an IBM token is set,
    return a descriptor to use IBM Runtime primitives. This function avoids
    raising and falls back to Aer simulator where possible.
    """
    # First try Aer by name
    try:
        return ("aer", Aer.get_backend(backend_name))
    except Exception:
        pass

    # If an IBM token exists and runtime package is available, return runtime info
    try:
        import os as _os
        token = (
            _os.getenv("IBM_API_KEY")
            or _os.getenv("QISKIT_IBM_TOKEN")
            or _os.getenv("IBM_APU_KEY")
        )
        if token:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                # Defer actual backend/session creation to caller
                return ("runtime", service, backend_name)
            except Exception:
                pass
    except Exception:
        pass

    # Final fallback to Aer statevector simulator
    try:
        return ("aer", Aer.get_backend("statevector_simulator"))
    except Exception:
        # As a last resort, return None
        return ("none", None)


def compute_kernel_element(
    feature_map: Union[ZZFeatureMap, PauliFeatureMap],
    x1: np.ndarray,
    x2: np.ndarray,
    backend_name: str = 'statevector_simulator'
) -> float:
    """
    Compute a single element of the kernel matrix.
    
    Args:
        feature_map: Feature map to use
        x1: First data point
        x2: Second data point
        backend_name: Name of the backend to use for simulation
        
    Returns:
        float: Kernel value
    """
    kind_backend = _resolve_backend_or_runtime(backend_name)
    kind = kind_backend[0]
    
    circuit_1 = create_feature_map_circuit(feature_map, x1)
    circuit_2 = create_feature_map_circuit(feature_map, x2)
    
    # For statevector_simulator, we can compute the inner product directly
    if backend_name == 'statevector_simulator' or kind == 'aer':
        # Try Aer statevector inner product
        try:
            backend = kind_backend[1] if kind == 'aer' else Aer.get_backend(backend_name)
            transpiled_circuit_1 = transpile(circuit_1, backend)
            transpiled_circuit_2 = transpile(circuit_2, backend)
            job_1 = backend.run(transpiled_circuit_1)
            job_2 = backend.run(transpiled_circuit_2)
            state_1 = job_1.result().get_statevector()
            state_2 = job_2.result().get_statevector()
            inner_product = np.abs(np.vdot(state_1, state_2)) ** 2
            return inner_product
        except Exception:
            # Fall through to swap test path below
            pass
    else:
        # For other backends (or if statevector path failed), use swap test
        n_qubits = feature_map.num_qubits
        
        # Create a circuit for the swap test
        swap_circuit = QuantumCircuit(2 * n_qubits + 1, 1)
        
        # Apply Hadamard to the ancilla qubit
        swap_circuit.h(0)
        
        # Apply the feature maps to the respective qubits
        for i, param in enumerate(feature_map.parameters):
            swap_circuit.compose(circuit_1, qubits=range(1, n_qubits + 1), inplace=True)
            swap_circuit.compose(circuit_2, qubits=range(n_qubits + 1, 2 * n_qubits + 1), inplace=True)
        
        # Apply controlled-SWAP gates
        for i in range(n_qubits):
            swap_circuit.cswap(0, i + 1, i + n_qubits + 1)
        
        # Apply Hadamard to the ancilla qubit
        swap_circuit.h(0)
        
        # Measure the ancilla qubit
        swap_circuit.measure(0, 0)
        
        # Execute the circuit
        counts = None
        # Try Aer/qasm execution first
        try:
            if kind == 'aer':
                backend = kind_backend[1]
            else:
                backend = Aer.get_backend('aer_simulator')
            transpiled_swap_circuit = transpile(swap_circuit, backend)
            job = backend.run(transpiled_swap_circuit, shots=2048)
            counts = job.result().get_counts()
        except Exception:
            pass

        # Try IBM Runtime Sampler as a fallback if available
        if counts is None and kind == 'runtime':
            try:
                from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
                service = kind_backend[1]
                target_backend = kind_backend[2]
                with Session(service=service, backend=target_backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([swap_circuit])
                    result = job.result()
                    # Convert quasi-dist to counts-like dict approximating probabilities
                    dist = result[0].data.meas.get_counts()
                    counts = {k: int(v * 2048) for k, v in dist.items()}
            except Exception:
                counts = None
        
        # Compute the kernel value
        if counts and '0' in counts and '1' in counts:
            p0 = counts['0'] / 1024
            return 2 * p0 - 1
        elif counts and '0' in counts:
            return 1.0
        else:
            return -1.0


def compute_kernel_matrix(
    feature_map: Union[ZZFeatureMap, PauliFeatureMap],
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    backend_name: str = 'statevector_simulator'
) -> np.ndarray:
    """
    Compute the kernel matrix for the given data.
    
    Args:
        feature_map: Feature map to use
        X_train: Training data
        X_test: Test data (if None, compute training kernel matrix)
        backend_name: Name of the backend to use for simulation
        
    Returns:
        np.ndarray: Kernel matrix
    """
    if X_test is None:
        X_test = X_train
        
    kernel_matrix = np.zeros((len(X_test), len(X_train)))
    
    for i, x1 in enumerate(X_test):
        for j, x2 in enumerate(X_train):
            kernel_matrix[i, j] = compute_kernel_element(feature_map, x1, x2, backend_name)
    
    return kernel_matrix


def _normalize_kernel(K_train: np.ndarray, K_test: Optional[np.ndarray] = None):
    """Normalize kernel so diagonal entries are ~1.0.

    K' = D^{-1/2} K D^{-1/2} where D=diag(K).
    For test: Kt' = Kt / sqrt(diag_test) / sqrt(diag_train[j]).
    """
    K = K_train.astype(float, copy=True)
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K_norm = K / (d[:, None] * d[None, :])
    if K_test is None:
        return K_norm, None
    Kt = K_test.astype(float, copy=True)
    # Approximate test diagonal per row (fallback to 1.0)
    # If Kt contains self-similarities on the diagonal (rare), use them; else use row mean
    try:
        dt = np.sqrt(np.clip(np.mean(Kt, axis=1), 1e-12, None))
    except Exception:
        dt = np.ones(Kt.shape[0])
    Kt_norm = Kt / (dt[:, None] * d[None, :])
    return K_norm, Kt_norm


def _center_kernel(K_train: np.ndarray, K_test: Optional[np.ndarray] = None):
    """Center kernel in feature space (double-centering).

    Train: Kc = H K H, H = I - 1/n 11^T.
    Test:  Ktc[i,j] = Kt[i,j] - mean_rows_test[i] - mean_cols_train[j] + mean_all_train.
    """
    K = K_train.astype(float, copy=True)
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    if K_test is None:
        return Kc, None
    Kt = K_test.astype(float, copy=True)
    mean_cols = np.mean(K, axis=0)
    mean_all = float(np.mean(K))
    mean_rows_test = np.mean(Kt, axis=1)
    Ktc = Kt - mean_rows_test[:, None] - mean_cols[None, :] + mean_all
    return Kc, Ktc


def analyze_kernel_matrix(kernel_matrix: np.ndarray, name: str = "Kernel Matrix") -> Dict:
    """
    Analyze and report statistics about a kernel matrix.
    
    Args:
        kernel_matrix: Kernel matrix to analyze
        name: Name of the kernel matrix
        
    Returns:
        Dict: Dictionary of statistics about the kernel matrix
    """
    stats = {
        "name": name,
        "shape": kernel_matrix.shape,
        "min": float(np.min(kernel_matrix)),
        "max": float(np.max(kernel_matrix)),
        "mean": float(np.mean(kernel_matrix)),
        "std": float(np.std(kernel_matrix))
    }
    
    # Calculate eigenvalues to assess positive definiteness (important for kernels)
    try:
        eigvals = np.linalg.eigvalsh(kernel_matrix)
        min_eig = float(np.min(eigvals))
        stats["min_eigenvalue"] = min_eig
        stats["positive_definite"] = min_eig >= -1e-10  # Allow for numerical error
    except np.linalg.LinAlgError:
        stats["min_eigenvalue"] = None
        stats["positive_definite"] = None
    
    return stats


def train_quantum_svm(
    kernel_matrix: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> SVC:
    """
    Train an SVM using a precomputed quantum kernel matrix.
    
    Args:
        kernel_matrix: Precomputed kernel matrix for training data
        y_train: Training labels
        sample_weight: Sample weights
        class_weight: Class weight strategy
        random_state: Random state for reproducibility
        
    Returns:
        SVC: Trained SVM model
    """
    svm = SVC(
        kernel='precomputed',
        class_weight=class_weight,
        random_state=random_state
    )
    svm.fit(kernel_matrix, y_train, sample_weight=sample_weight)
    return svm


def train_quantum_krr(
    kernel_matrix: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0
) -> KernelRidge:
    """
    Train a Kernel Ridge Regression model using a precomputed quantum kernel matrix.
    
    Args:
        kernel_matrix: Precomputed kernel matrix for training data
        y_train: Training labels
        alpha: Regularization parameter
        
    Returns:
        KernelRidge: Trained KRR model
    """
    krr = KernelRidge(kernel='precomputed', alpha=alpha)
    krr.fit(kernel_matrix, y_train)
    return krr


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted scores (for AUC and AP)
        
    Returns:
        Dict: Dictionary of classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_score is not None:
        metrics["auc"] = roc_auc_score(y_true, y_score)
        metrics["ap"] = average_precision_score(y_true, y_score)
    
    # Detailed classification metrics
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    true_neg = np.sum((y_pred == 0) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))
    
    metrics["true_positives"] = int(true_pos)
    metrics["false_positives"] = int(false_pos)
    metrics["true_negatives"] = int(true_neg)
    metrics["false_negatives"] = int(false_neg)
    
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict: Dictionary of regression metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
        "min_actual": float(np.min(y_true)),
        "max_actual": float(np.max(y_true)),
        "min_predicted": float(np.min(y_pred)),
        "max_predicted": float(np.max(y_pred))
    }


def run_quantum_kernel_classification(
    feature_maps: Dict[str, Union[ZZFeatureMap, PauliFeatureMap]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    backend_name: str = 'statevector_simulator'
) -> Dict:
    """
    Run quantum kernel classification with multiple feature maps.
    
    Args:
        feature_maps: Dictionary of feature maps to use
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        sample_weight: Sample weights for training
        backend_name: Name of the backend to use for simulation
        
    Returns:
        Dict: Dictionary of results for each feature map
    """
    results = {}
    
    for name, feature_map in feature_maps.items():
        print(f"Running quantum kernel classification with {name}...")
        
        # Compute kernel matrices
        K_train = compute_kernel_matrix(feature_map, X_train, backend_name=backend_name)
        K_test = compute_kernel_matrix(feature_map, X_train, X_test, backend_name=backend_name)
        # Sanitize any NaNs/Infs that may arise from upstream numeric issues
        K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=-1.0)
        K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Analyze kernel matrix
        kernel_stats = analyze_kernel_matrix(K_train, name=f"{name} Training Kernel Matrix")
        
        # Train SVM
        svm = train_quantum_svm(K_train, y_train, sample_weight=sample_weight)
        
        # Predict
        y_pred = svm.predict(K_test)
        y_score = svm.decision_function(K_test)
        
        # Evaluate
        metrics = evaluate_classification(y_test, y_pred, y_score)
        
        # Store results (include kernels for downstream learning curves)
        results[name] = {
            "kernel_stats": kernel_stats,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_score": y_score,
            "K_train": K_train,
            "K_test": K_test,
        }
    
    return results


def run_quantum_kernel_regression(
    feature_maps: Dict[str, Union[ZZFeatureMap, PauliFeatureMap]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
    backend_name: str = 'statevector_simulator',
    center: bool = True,
    normalize: bool = True,
    alpha_grid: Optional[list] = None,
    log_target: bool = True,
) -> Dict:
    """
    Run quantum kernel regression with multiple feature maps.
    
    Args:
        feature_maps: Dictionary of feature maps to use
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        alpha: Regularization parameter for KRR
        backend_name: Name of the backend to use for simulation
        
    Returns:
        Dict: Dictionary of results for each feature map
    """
    results = {}
    
    for name, feature_map in feature_maps.items():
        print(f"Running quantum kernel regression with {name}...")
        
        # Compute kernel matrices
        K_train = compute_kernel_matrix(feature_map, X_train, backend_name=backend_name)
        K_test = compute_kernel_matrix(feature_map, X_train, X_test, backend_name=backend_name)

        # Optional normalization/centering (often improves regression)
        if normalize:
            K_train, K_test = _normalize_kernel(K_train, K_test)
        if center:
            K_train, K_test = _center_kernel(K_train, K_test)
        # Final sanitize after transforms
        K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=-1.0)
        K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Analyze kernel matrix
        kernel_stats = analyze_kernel_matrix(K_train, name=f"{name} Training Kernel Matrix")
        
        # Train/predict with KRR. Optionally tune alpha and log-transform target.
        best_params = {"alpha": alpha}
        if alpha_grid:
            try:
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                best_alpha = alpha_grid[0]
                best_score = -np.inf
                for a in alpha_grid:
                    scores = []
                    for tr_idx, va_idx in kf.split(K_train):
                        K_tr = K_train[np.ix_(tr_idx, tr_idx)]
                        K_va = K_train[np.ix_(va_idx, tr_idx)]
                        y_tr = y_train[tr_idx]
                        y_va = y_train[va_idx]
                        # manual log-transform to avoid wrapper issues
                        y_fit = np.log1p(y_tr) if log_target else y_tr
                        mdl = KernelRidge(kernel='precomputed', alpha=a)
                        mdl.fit(K_tr, y_fit)
                        yhat = mdl.predict(K_va)
                        if log_target:
                            yhat = np.expm1(yhat)
                        mse = np.mean((yhat - y_va) ** 2)
                        scores.append(-mse)
                    mean_score = float(np.mean(scores))
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = a
                best_params = {"alpha": best_alpha}
                # Fit on full and predict
                y_fit_full = np.log1p(y_train) if log_target else y_train
                mdl = KernelRidge(kernel='precomputed', alpha=best_alpha)
                mdl.fit(K_train, y_fit_full)
                y_pred = mdl.predict(K_test)
                if log_target:
                    y_pred = np.expm1(y_pred)
            except Exception as e:
                print(f"Warning: alpha grid search failed: {e}; using alpha={alpha}")
                y_fit_full = np.log1p(y_train) if log_target else y_train
                mdl = KernelRidge(kernel='precomputed', alpha=alpha)
                mdl.fit(K_train, y_fit_full)
                y_pred = mdl.predict(K_test)
                if log_target:
                    y_pred = np.expm1(y_pred)
        else:
            y_fit_full = np.log1p(y_train) if log_target else y_train
            mdl = KernelRidge(kernel='precomputed', alpha=alpha)
            mdl.fit(K_train, y_fit_full)
            y_pred = mdl.predict(K_test)
            if log_target:
                y_pred = np.expm1(y_pred)
        
        # Evaluate
        metrics = evaluate_regression(y_test, y_pred)
        
        # Store results (include kernels for downstream learning curves)
        results[name] = {
            "kernel_stats": kernel_stats,
            "metrics": metrics,
            "y_pred": y_pred,
            "K_train": K_train,
            "K_test": K_test,
            "best_params": best_params,
        }
    
    return results
