"""
Convenience runner for end-to-end pipelines (occurrence + severity).

Usage:
  python quantum/run_end_to_end.py

Optional: edit the flags below to toggle quantum learning curves or SVM calibration.
"""

import os
import sys

# Ensure package imports work when running this file directly:
# add the project root to sys.path so `import quantum.*` resolves.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum.enhanced_main import run_enhanced_quantum_solution
from quantum.run_severity_analysis import run_severity_analysis


def main():
    data_path = "data/mcc.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # End-to-end: occurrence (binary classification) + severity (regression)
    run_enhanced_quantum_solution(
        data_path=data_path,
        k_qubits=8,
        backend_name="qasm_simulator",
        results_dir=results_dir,
        focus_on_severity=True,
        compute_quantum_learning_curves=False,  # set True if you want quantum LCs
        quantum_lc_max_n=800,
        calibrate_svm=True,
    )

    # Optional: severity-only classical baseline (additional diagnostics)
    run_severity_analysis(data_path=data_path, results_dir=results_dir)


if __name__ == "__main__":
    main()
