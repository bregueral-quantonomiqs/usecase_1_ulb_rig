#!/usr/bin/env python
"""
Run severity claim analysis using IBM Qiskit (quantum kernel regression only).

This script focuses purely on the severity (regression) component and uses
Qiskit Aer (statevector) to compute quantum kernel matrices. It allows
configurable train/test sizes to keep runtimes manageable.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict
import sys
import pathlib

import numpy as np
import warnings

# Ensure project root is on path for `quantum.*` imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from quantum.enhanced_data_encoding import (
    load_clean,
    enhanced_quantum_ready_splits,
)
from quantum.feature_maps import get_all_feature_maps
from quantum.kernel_methods import run_quantum_kernel_regression
from quantum.visualization import visualize_gate_level_circuit
from quantum.config import load_ibm_token


def run_severity_qiskit(
    data_path: str = "data/mcc.csv",
    k_qubits: int = 6,
    backend_name: str = "statevector_simulator",
    n_train: int = 60,
    n_test: int = 20,
    results_dir: str = "results",
    feature_map_names: list[str] | None = None,
) -> Dict:
    """Run Qiskit-based quantum kernel regression for severity only.

    Args:
        data_path: Path to CSV dataset.
        k_qubits: Number of qubits for feature map dimension.
        backend_name: Qiskit backend name (e.g. 'statevector_simulator').
        n_train: Number of training samples for severity.
        n_test: Number of test samples for severity.
        results_dir: Directory to save results.

    Returns:
        Dict with results per feature map (metrics and predictions).
    """
    # Normalize results_dir to project root to avoid CWD surprises
    if not os.path.isabs(results_dir):
        results_dir = str((ROOT / results_dir).resolve())
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data: {data_path}")
    # Load IBM token from config if not already in env
    load_ibm_token()
    df = load_clean(data_path)

    print(
        f"Preparing enhanced quantum-ready splits for severity (n_train={n_train}, n_test={n_test}, k={k_qubits})"
    )
    data = enhanced_quantum_ready_splits(
        df,
        k_qubits=k_qubits,
        augment_severity=False,  # keep core pipeline fast and deterministic
        n_train_sev=n_train,
        n_test_sev=n_test,
    )

    (_, _, _, _, _, _) = data["occ"]  # unused, but unpack to document shape
    (X_train_sev, y_train_sev, X_test_sev, y_test_sev) = data["sev"]

    # Safety: ensure shapes are 2D arrays for circuits
    X_train_sev = np.asarray(X_train_sev)
    X_test_sev = np.asarray(X_test_sev)
    y_train_sev = np.asarray(y_train_sev)
    y_test_sev = np.asarray(y_test_sev)

    if X_train_sev.ndim == 1:
        X_train_sev = X_train_sev.reshape(-1, 1)
    if X_test_sev.ndim == 1:
        X_test_sev = X_test_sev.reshape(-1, 1)

    print("Building feature maps...")
    all_maps = get_all_feature_maps(feature_dimension=k_qubits, reps=2)
    if feature_map_names:
        # robust, case-insensitive selection
        key_map = {k.lower(): k for k in all_maps.keys()}
        selected = {}
        for nm in feature_map_names:
            key = nm.lower()
            if key in key_map:
                selected[key_map[key]] = all_maps[key_map[key]]
        feature_maps = selected
        if not feature_maps:
            raise ValueError(f"No matching feature maps found in {list(all_maps.keys())}")
    else:
        # Default to a single fast map to keep runtime manageable
        feature_maps = {"zz_linear": all_maps["zz_linear"]}

    print(
        f"Running quantum kernel regression on backend '{backend_name}' with {len(feature_maps)} feature map(s)..."
    )
    results = run_quantum_kernel_regression(
        feature_maps,
        X_train_sev,
        y_train_sev,
        X_test_sev,
        y_test_sev,
        backend_name=backend_name,
        center=True,
        normalize=True,
        alpha_grid=[0.01, 0.1, 1.0, 10.0, 100.0],
        log_target=True,
    )

    # Gate-level visualization (IBM style) for selected maps using first train sample as binding
    try:
        x_sample = X_train_sev[0]
        # Attempt IBM Runtime backend for transpilation if available
        ibm_backend = None
        try:
            import os as _os
            token = (
                _os.getenv("IBM_API_KEY")
                or _os.getenv("QISKIT_IBM_TOKEN")
                or _os.getenv("IBM_APU_KEY")  # tolerate user-provided var name
            )
            if token:
                from qiskit_ibm_runtime import QiskitRuntimeService

                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                # If user passed an IBM backend name, try to fetch it; otherwise skip
                if backend_name and not backend_name.startswith("statevector") and not backend_name.startswith("aer_"):
                    try:
                        ibm_backend = service.backend(backend_name)
                    except Exception:
                        ibm_backend = None
        except Exception:
            ibm_backend = None

        for name, fmap in feature_maps.items():
            try:
                # Bind parameters for a concrete circuit snapshot
                params_dict = {p: x_sample[i] for i, p in enumerate(fmap.parameters)}
                bound = fmap.assign_parameters(params_dict)
                prefix = f"severity_{name}"
                visualize_gate_level_circuit(
                    bound,
                    filename_prefix=prefix,
                    results_dir=results_dir,
                    backend=ibm_backend,
                )
            except Exception as e:
                print(f"Gate-level visualization failed for {name}: {e}")
    except Exception as e:
        print(f"Gate-level visualization skipped: {e}")

    # Persist a compact metrics summary
    out_path = os.path.join(results_dir, "severity_qiskit_metrics.txt")
    try:
        with open(out_path, "w") as f:
            f.write("Qiskit Severity Regression (Quantum Kernel)\n")
            f.write("==========================================\n\n")
            f.write(f"Data: {data_path}\n")
            f.write(f"Qubits: {k_qubits}\n")
            f.write(f"Backend: {backend_name}\n")
            f.write(f"Train/Test: {len(X_train_sev)}/{len(X_test_sev)}\n\n")
            for name, res in results.items():
                m = res.get("metrics", {})
                f.write(f"[{name}] MAE={m.get('mae'):.4f} RMSE={m.get('rmse'):.4f} R2={m.get('r2'):.4f}\n")
        print(f"Saved metrics to {out_path}")
    except Exception as e:
        print(f"Warning: failed to write metrics to {out_path}: {e}")
        print("Metrics summary:")
        for name, res in results.items():
            m = res.get("metrics", {})
            print(f"[{name}] MAE={m.get('mae'):.4f} RMSE={m.get('rmse'):.4f} R2={m.get('r2'):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Qiskit severity quantum kernel regression")
    parser.add_argument("--data-path", type=str, default="data/mcc.csv")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--backend", type=str, default="statevector_simulator")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--feature-maps",
        type=str,
        default="zz_linear",
        help="Comma-separated list of feature map names (e.g., zz_linear,pauli_linear)",
    )
    args = parser.parse_args()

    print("Configuration:")
    print(f"  data_path:   {args.data_path}")
    print(f"  qubits:      {args.qubits}")
    print(f"  backend:     {args.backend}")
    print(f"  n_train:     {args.n_train}")
    print(f"  n_test:      {args.n_test}")
    print(f"  results_dir: {args.results_dir}")

    fm = [s.strip().lower() for s in args.feature_maps.split(",") if s.strip()]
    run_severity_qiskit(
        data_path=args.data_path,
        k_qubits=args.qubits,
        backend_name=args.backend,
        n_train=args.n_train,
        n_test=args.n_test,
        results_dir=args.results_dir,
        feature_map_names=fm,
    )


if __name__ == "__main__":
    main()
    # Quiet safe, known warnings from sklearn about feature names and pre-fit pipeline
    warnings.filterwarnings(
        "ignore",
        message=r"X has feature names, but QuantileTransformer was fitted without feature names",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"This Pipeline instance is not fitted yet.*",
        category=FutureWarning,
    )
# Silence benign sklearn warnings (fit-before-transform deprecation and feature-name noise)
warnings.filterwarnings(
    "ignore",
    message=r"This Pipeline instance is not fitted yet.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but QuantileTransformer was fitted without feature names",
    category=UserWarning,
)
