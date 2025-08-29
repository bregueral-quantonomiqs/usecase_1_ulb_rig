#!/usr/bin/env python
"""
Run occurrence (classification) analysis using IBM Qiskit (quantum kernel SVM).

This script focuses on the classification problem (occur: 0/1) and uses
Qiskit Aer (statevector by default) to compute quantum kernel matrices. It
also supports IBM gate-level visualization via transpilation, and will target
an IBM backend for transpilation if a valid token + runtime package is present.
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

from quantum.enhanced_data_encoding import (
    load_clean,
    enhanced_quantum_ready_splits,
)
from quantum.feature_maps import get_all_feature_maps
from quantum.kernel_methods import run_quantum_kernel_classification
from quantum.visualization import visualize_gate_level_circuit
from quantum.config import load_ibm_token


def run_occurrence_qiskit(
    data_path: str = "data/mcc.csv",
    k_qubits: int = 6,
    backend_name: str = "statevector_simulator",
    n_train: int = 120,
    n_test: int = 40,
    results_dir: str = "results",
    feature_map_names: list[str] | None = None,
) -> Dict:
    """Run Qiskit-based quantum kernel SVM for occurrence classification.

    Args:
        data_path: Path to CSV dataset.
        k_qubits: Number of qubits for feature map dimension.
        backend_name: Qiskit backend name (e.g. 'statevector_simulator').
        n_train: Number of training samples.
        n_test: Number of test samples.
        results_dir: Directory to save results.
        feature_map_names: Optional subset of feature maps to run.

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
        f"Preparing enhanced quantum-ready splits for occurrence (n_train={n_train}, n_test={n_test}, k={k_qubits})"
    )
    data = enhanced_quantum_ready_splits(
        df,
        k_qubits=k_qubits,
        augment_severity=False,
        n_train_occ=n_train,
        n_test_occ=n_test,
        # keep default severity sizes; we only use occurrence here
    )

    (X_train_occ, y_train_occ, w_train_occ, X_test_occ, y_test_occ, w_test_occ) = data["occ"]

    # Ensure 2D arrays
    X_train_occ = np.asarray(X_train_occ)
    X_test_occ = np.asarray(X_test_occ)
    y_train_occ = np.asarray(y_train_occ)
    y_test_occ = np.asarray(y_test_occ)
    if X_train_occ.ndim == 1:
        X_train_occ = X_train_occ.reshape(-1, 1)
    if X_test_occ.ndim == 1:
        X_test_occ = X_test_occ.reshape(-1, 1)

    print("Building feature maps...")
    all_maps = get_all_feature_maps(feature_dimension=k_qubits, reps=2)
    if feature_map_names:
        feature_maps = {k: v for k, v in all_maps.items() if k in feature_map_names}
        if not feature_maps:
            raise ValueError(f"No matching feature maps found in {list(all_maps.keys())}")
    else:
        feature_maps = {"zz_linear": all_maps["zz_linear"]}

    print(
        f"Running quantum kernel classification on backend '{backend_name}' with {len(feature_maps)} feature map(s)..."
    )
    results = run_quantum_kernel_classification(
        feature_maps,
        X_train_occ,
        y_train_occ,
        X_test_occ,
        y_test_occ,
        sample_weight=w_train_occ,
        backend_name=backend_name,
    )

    # Save metrics summary
    out_path = os.path.join(results_dir, "occurrence_qiskit_metrics.txt")
    try:
        with open(out_path, "w") as f:
            f.write("Qiskit Occurrence Classification (Quantum Kernel SVM)\n")
            f.write("====================================================\n\n")
            f.write(f"Data: {data_path}\n")
            f.write(f"Qubits: {k_qubits}\n")
            f.write(f"Backend: {backend_name}\n")
            f.write(f"Train/Test: {len(X_train_occ)}/{len(X_test_occ)}\n\n")
            for name, res in results.items():
                m = res.get("metrics", {})
                f.write(
                    f"[{name}] AUC={m.get('auc', float('nan')):.4f} F1={m.get('f1', float('nan')):.4f} Acc={m.get('accuracy', float('nan')):.4f}\n"
                )
        print(f"Saved metrics to {out_path}")
    except Exception as e:
        print(f"Warning: failed to write metrics to {out_path}: {e}")
        print("Metrics summary:")
        for name, res in results.items():
            m = res.get("metrics", {})
            print(
                f"[{name}] AUC={m.get('auc', float('nan')):.4f} F1={m.get('f1', float('nan')):.4f} Acc={m.get('accuracy', float('nan')):.4f}"
            )

    # Gate-level visualization (IBM style)
    try:
        x_sample = X_train_occ[0]
        ibm_backend = None
        try:
            import os as _os
            token = (
                _os.getenv("IBM_API_KEY")
                or _os.getenv("QISKIT_IBM_TOKEN")
                or _os.getenv("IBM_APU_KEY")
            )
            if token:
                from qiskit_ibm_runtime import QiskitRuntimeService

                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                if backend_name and not backend_name.startswith("statevector") and not backend_name.startswith("aer_"):
                    try:
                        ibm_backend = service.backend(backend_name)
                    except Exception:
                        ibm_backend = None
        except Exception:
            ibm_backend = None

        for name, fmap in feature_maps.items():
            try:
                params_dict = {p: x_sample[i] for i, p in enumerate(fmap.parameters)}
                bound = fmap.assign_parameters(params_dict)
                prefix = f"occurrence_{name}"
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

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Qiskit occurrence quantum kernel classification")
    parser.add_argument("--data-path", type=str, default="data/mcc.csv")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--backend", type=str, default="statevector_simulator")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--feature-maps",
        type=str,
        default="zz_linear",
        help="Comma-separated feature map names (e.g., zz_linear,pauli_linear)",
    )
    args = parser.parse_args()

    print("Configuration:")
    print(f"  data_path:   {args.data_path}")
    print(f"  qubits:      {args.qubits}")
    print(f"  backend:     {args.backend}")
    print(f"  n_train:     {args.n_train}")
    print(f"  n_test:      {args.n_test}")
    print(f"  results_dir: {args.results_dir}")

    fm = [s.strip() for s in args.feature_maps.split(",") if s.strip()]
    run_occurrence_qiskit(
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
