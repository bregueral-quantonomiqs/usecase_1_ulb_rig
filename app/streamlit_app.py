"""
Streamlit demo app for the insurance use case.

Features:
- Configure dataset path, number of qubits, backend, and runtime options
- Show encoded "initial angles" (first training sample) for occurrence and severity
- Run the end-to-end pipeline with progress indicators
- Display metrics and visualization images
- Detect IBM token; if available, the kernels attempt to use IBM Runtime automatically

Run locally:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

import streamlit as st

# Ensure project root on path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum.config import load_ibm_token
from quantum.enhanced_main import run_enhanced_quantum_solution
from quantum.feature_maps import get_all_feature_maps
from quantum.visualization import visualize_circuit, visualize_gate_level_circuit
from quantum.enhanced_data_encoding import load_clean, enhanced_quantum_ready_splits


RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA = PROJECT_ROOT / "data" / "mcc.csv"


def list_result_images(pattern_prefixes: List[str]) -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    images: List[Path] = []
    for p in RESULTS_DIR.glob("*.png"):
        if any(p.name.startswith(pref) for pref in pattern_prefixes):
            images.append(p)
    return sorted(images)


def load_results_json() -> Dict:
    p = RESULTS_DIR / "enhanced_quantum_results.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


LOGO_PATH = PROJECT_ROOT / "app" / "assets" / "logo.png"
page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else None
st.set_page_config(page_title="Quantum Insurance Demo", layout="wide", page_icon=page_icon)

if LOGO_PATH.exists():
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.image(str(LOGO_PATH), width='stretch')
else:
    st.title("Quantum Insurance Demo")

@st.cache_data(show_spinner=False)
def cached_ibm_backends(min_qubits: int = 1) -> list[str]:
    """
    Return available IBM real device names with at least `min_qubits` qubits.
    Filters out simulators and, where possible, offline devices.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum")
        devices = []
        for b in svc.backends():
            try:
                # Filter out simulators
                if getattr(b, 'simulator', False):
                    continue
                n = int(getattr(b, 'num_qubits', 0) or 0)
                if n < int(min_qubits):
                    continue
                # Prefer online devices if status available
                ok = True
                try:
                    stt = getattr(b, 'status', None)
                    if callable(stt):
                        stt = stt()
                    # Qiskit status objects may have .status or .operational
                    if stt is not None:
                        if hasattr(stt, 'operational') and not stt.operational:
                            ok = False
                        if hasattr(stt, 'status') and hasattr(stt.status, 'value'):
                            # If explicitly not ONLINE
                            if str(stt.status).lower() not in ('online', 'Status.ONLINE'.lower()):
                                ok = False
                except Exception:
                    pass
                if ok:
                    devices.append(b.name)
            except Exception:
                continue
        return sorted(set(devices))
    except Exception:
        # Fallback list (may include unavailable devices)
        return [
            "ibm_quito", "ibm_nairobi", "ibm_oslo", "ibm_perth",
            "ibm_tokyo", "ibm_kyoto", "ibm_osaka",
        ]


with st.sidebar:
    st.header("Configuration")

    # Data selection / upload
    data_path_str = st.text_input("Data path", str(DEFAULT_DATA))
    uploaded = st.file_uploader("Or upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        up_dst = PROJECT_ROOT / "data" / "uploaded.csv"
        up_dst.parent.mkdir(parents=True, exist_ok=True)
        up_dst.write_bytes(uploaded.read())
        data_path_str = str(up_dst)
        st.caption(f"Uploaded dataset saved to {up_dst}")

    # Problem selection / task
    task_choice = st.selectbox(
        "Problem",
        options=["Both", "Occurrence (classification)", "Severity (regression)"],
        index=0,
        help="Choose which problem to focus on in the UI"
    )

    # Run mode presets
    run_mode = st.selectbox(
        "Run Mode",
        options=["Fast", "Medium", "Full"],
        index=1,
        help="Fast = tiny/quick demo, Medium = balanced, Full = thorough (slow)"
    )

    k_qubits = st.slider("Number of qubits (controls overridden by mode)", min_value=2, max_value=16, value=8, step=1)

    st.subheader("Backend")
    # Accept token from Streamlit secrets or manual input (fallback), then load from config/env
    try:
        if "IBM_API_KEY" in st.secrets and not os.getenv("IBM_API_KEY"):
            os.environ["IBM_API_KEY"] = st.secrets["IBM_API_KEY"]
    except Exception:
        pass
    token_present = bool(load_ibm_token())
    backend_category = st.selectbox(
        "Select backend type",
        options=["Simulators", "IBM Real"],
        index=0,
        help="Choose a local/cloud simulator or a real IBM device",
    )
    sim_options = [
        "qasm_simulator",            # Aer QASM simulator
        "statevector_simulator",     # Aer statevector simulator
        "ibm_qasm_simulator",        # IBM cloud simulator
    ]
    if backend_category == "Simulators":
        selected_backend = st.selectbox("Simulator backend", options=sim_options, index=0)
    else:
        if not token_present:
            st.warning("IBM token not detected. Switch to Simulators or paste an API key below.")
            selected_backend = st.selectbox("IBM real device (disabled)", options=["-- no token --"], index=0, disabled=True)
        else:
            ibm_list = cached_ibm_backends(int(k_qubits))
            if not ibm_list:
                st.error(f"No real IBM device found with ≥ {int(k_qubits)} qubits. Reduce qubits or use a simulator.")
            selected_backend = st.selectbox(
                "IBM real device",
                options=ibm_list or ["ibm_qasm_simulator"],
                index=0,
            )
    manual_override = st.text_input(
        "Manual backend override (optional)",
        value="",
        help="Enter a custom backend name; if set, this overrides the selection above.",
    )
    backend_name = manual_override.strip() or selected_backend

    st.subheader("Options")
    compute_q_lc = st.checkbox("Compute quantum learning curves", value=False)
    q_lc_max_n = st.number_input("Quantum LC max n", min_value=100, max_value=5000, value=800, step=50)
    calibrate_svm = st.checkbox("Calibrate SVM probabilities (sigmoid)", value=True)
    skip_classical = st.checkbox("Skip classical baselines", value=True, help="Only run quantum methods (faster)")

    st.subheader("Feature Maps (controls overridden by mode)")
    fm_all = [
        "zz_linear", "zz_circular", "zz_full",
        "pauli_linear", "pauli_circular", "pauli_full",
    ]
    selected_fms = st.multiselect(
        "Select feature maps",
        options=fm_all,
        default=["zz_circular"],
        help="Subset of feature maps to use for quantum kernels",
    )

    st.subheader("Train/Test Sizes (controls overridden by mode)")
    n_train_occ = st.number_input("Occurrence: n_train", min_value=20, max_value=5000, value=60, step=10)
    n_test_occ = st.number_input("Occurrence: n_test", min_value=10, max_value=2000, value=20, step=10)
    n_train_sev = st.number_input("Severity: n_train", min_value=20, max_value=5000, value=60, step=10)
    n_test_sev = st.number_input("Severity: n_test", min_value=10, max_value=2000, value=20, step=10)

    st.subheader("IBM Runtime")
    st.write("IBM token detected:" if token_present else "IBM token not detected", token_present)
    if not token_present:
        pasted = st.text_input("IBM API Key (optional)", type="password")
        if pasted:
            os.environ["IBM_API_KEY"] = pasted
            token_present = True
            st.success("IBM API key set for this session.")
    st.caption("Provide IBM_API_KEY via environment, config/ibm_config.json, or Streamlit secrets.")

    run_btn = st.button("Run Pipeline", type="primary")
    show_preview = st.checkbox("Show circuit preview before running", value=True)

@st.cache_data(show_spinner=False)
def cached_load_clean(path: str):
    return load_clean(path)


@st.cache_data(show_spinner=False)
def cached_splits(path: str, k_qubits: int):
    """
    Return only picklable parts of the enhanced splits for Streamlit caching.
    We intentionally drop Pipeline objects (which contain local classes/closures)
    to avoid pickle errors like "Can't get local object ... SafeQuantileTransformer".
    """
    df_local = load_clean(path)
    data = enhanced_quantum_ready_splits(df_local, k_qubits=k_qubits, augment_severity=False)
    # Keep only arrays/metrics that are safe to cache
    trimmed = {
        "occ": data.get("occ"),
        "sev": data.get("sev"),
        "feature_importance": data.get("feature_importance"),
    }
    return trimmed


st.write("### Encoded Initial Angles (first sample)")
col1, col2 = st.columns(2)

try:
    df = cached_load_clean(data_path_str)
    enc = cached_splits(data_path_str, int(k_qubits))
    (X_train_occ, y_train_occ, *_rest_occ) = enc["occ"]
    (X_train_sev, y_train_sev, *_rest_sev) = enc["sev"]
    with col1:
        st.write("Occurrence:")
        occ0 = X_train_occ[0] if isinstance(X_train_occ, (list, tuple)) else getattr(X_train_occ, "__array__", lambda: X_train_occ)()[0]
        st.code(str(occ0))
        try:
            occ0 = np.asarray(occ0).reshape(-1)
            df_occ = pd.DataFrame({"Angle": np.arange(len(occ0)), "Value": occ0})
            st.bar_chart(df_occ, x="Angle", y="Value")
        except Exception:
            pass
    with col2:
        st.write("Severity:")
        sev0 = X_train_sev[0] if isinstance(X_train_sev, (list, tuple)) else getattr(X_train_sev, "__array__", lambda: X_train_sev)()[0]
        st.code(str(sev0))
        try:
            sev0 = np.asarray(sev0).reshape(-1)
            df_sev = pd.DataFrame({"Angle": np.arange(len(sev0)), "Value": sev0})
            st.bar_chart(df_sev, x="Angle", y="Value")
        except Exception:
            pass
    # Optional circuit preview (based on selected feature maps and problem focus)
    if show_preview:
        try:
            st.subheader("Circuits (preview)")
            sample = None
            if task_choice == "Severity (regression)" and 'sev0' in locals():
                sample = np.asarray(sev0).reshape(-1)
            elif task_choice == "Occurrence (classification)" and 'occ0' in locals():
                sample = np.asarray(occ0).reshape(-1)
            else:
                sample = np.asarray(sev0 if 'sev0' in locals() else occ0).reshape(-1)

            # Build maps (filter to selected)
            all_maps = get_all_feature_maps(feature_dimension=int(k_qubits), reps=1)
            name_map = {k.lower(): k for k in all_maps.keys()}
            fm_use = selected_fms or ["zz_circular"]
            chosen = {}
            for nm in fm_use:
                key = nm.lower()
                if key in name_map:
                    chosen[name_map[key]] = all_maps[name_map[key]]

            # Generate images
            previews = []
            for name, fmap in chosen.items():
                try:
                    if sample is not None and len(sample) >= len(fmap.parameters):
                        params = {p: sample[i] for i, p in enumerate(fmap.parameters)}
                        bound = fmap.assign_parameters(params)
                        visualize_gate_level_circuit(bound, filename_prefix=f"preview_{name}", results_dir=str(RESULTS_DIR), backend=None)
                        previews.append(RESULTS_DIR / f"preview_{name}_gate_level.png")
                except Exception:
                    pass

            # Show previews
            for p in previews:
                if p.exists():
                    st.image(str(p), caption=p.name, width='stretch')
        except Exception as e:
            st.info(f"Circuit preview unavailable: {e}")
except Exception as e:
    st.warning(f"Could not preview initial angles: {e}")

st.write("---")

if run_btn:
    with st.spinner("Running end-to-end pipeline... this may take a few minutes"):
        try:
            # Apply run-mode presets (override controls for reproducible profiles)
            fm_all = [
                "zz_linear", "zz_circular", "zz_full",
                "pauli_linear", "pauli_circular", "pauli_full",
            ]
            if run_mode == "Fast":
                # Respect user's feature-map selection; default to zz_linear if none selected
                fm_use = selected_fms if selected_fms else ["zz_linear"]
                args = dict(
                    k_qubits=4,
                    n_train_occ=60, n_test_occ=20,
                    n_train_sev=60, n_test_sev=20,
                    compute_quantum_learning_curves=False,
                    calibrate_svm=False,
                    lightweight=True,
                    skip_visuals=True,
                    run_variational=False,
                    cv_folds=1,
                    small_grids=True,
                    skip_classical=True,
                )
            elif run_mode == "Full":
                fm_use = fm_all
                args = dict(
                    k_qubits=max(8, int(k_qubits)),
                    n_train_occ=1000, n_test_occ=400,
                    n_train_sev=1000, n_test_sev=400,
                    compute_quantum_learning_curves=True,
                    calibrate_svm=True,
                    lightweight=False,
                    skip_visuals=False,
                    run_variational=True,
                    cv_folds=3,
                    small_grids=False,
                )
            else:  # Medium
                fm_use = selected_fms or ["zz_circular"]
                args = dict(
                    k_qubits=int(k_qubits),
                    n_train_occ=600, n_test_occ=200,
                    n_train_sev=600, n_test_sev=200,
                    compute_quantum_learning_curves=False,
                    calibrate_svm=True,
                    lightweight=False,
                    skip_visuals=False,
                    run_variational=False,
                    cv_folds=2,
                    small_grids=False,
                    skip_classical=bool(skip_classical),
                )

            run_enhanced_quantum_solution(
                data_path=data_path_str,
                k_qubits=int(args["k_qubits"]),
                backend_name=backend_name,
                results_dir=str(RESULTS_DIR),
                focus_on_severity=True,
                compute_quantum_learning_curves=bool(args["compute_quantum_learning_curves"]),
                quantum_lc_max_n=int(q_lc_max_n),
                calibrate_svm=bool(args["calibrate_svm"]),
                feature_map_names=fm_use,
                n_train_occ=int(args["n_train_occ"]),
                n_test_occ=int(args["n_test_occ"]),
                n_train_sev=int(args["n_train_sev"]),
                n_test_sev=int(args["n_test_sev"]),
                run_occurrence=(task_choice in ("Both", "Occurrence (classification)")),
                run_severity=(task_choice in ("Both", "Severity (regression)")),
                lightweight=bool(args["lightweight"]),
                skip_visuals=bool(args["skip_visuals"]),
                run_variational=bool(args["run_variational"]),
                cv_folds=int(args["cv_folds"]),
                small_grids=bool(args["small_grids"]),
                skip_classical=bool(args.get("skip_classical", skip_classical)),
            )
            st.success("Pipeline finished")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    # Show metrics (trim to quantum severity metrics: MAE, MSE, RMSE, R2)
    results = load_results_json()
    if results:
        st.write("### Results (quantum metrics)")
        trimmed = {}
        # Consider quantum_kernel and variational buckets only
        qk = results.get("quantum_kernel", {})
        var = results.get("variational", {})
        # Normalize selected feature map names to lower
        selected_lower = {s.lower() for s in (selected_fms or [])}
        def keep_map(name: str) -> bool:
            if not selected_lower:
                return True
            return any(sel in name.lower() for sel in selected_lower)
        # Severity regression metrics only
        for name, data in {**qk, **var}.items():
            if not keep_map(name):
                continue
            m = data.get("metrics", {})
            mae = m.get("mae"); mse = m.get("mse"); rmse = m.get("rmse"); r2 = m.get("r2")
            # Skip dummies (all zeros or None)
            if all(v in (0, 0.0, None) for v in (mae, mse, rmse, r2)):
                continue
            trimmed[name] = {k: v for k, v in (('mae', mae), ('mse', mse), ('rmse', rmse), ('r2', r2)) if v is not None}
        if trimmed:
            st.json(trimmed)
        else:
            st.info("No quantum severity metrics found (yet).")
    else:
        st.info("No results JSON found. Check logs.")

    # Show images — PNG only: gate-level circuits and quantum results (no classical)
    st.write("### Visualizations")
    # Prefix filter by problem
    if task_choice == "Occurrence (classification)":
        prefixes = ["enhanced_"]
    elif task_choice == "Severity (regression)":
        prefixes = ["enhanced_", "severity_"]
    else:
        prefixes = ["enhanced_", "severity_"]
    imgs = list_result_images(prefixes)
    # Filter by selected feature maps
    sel = {s.lower() for s in (selected_fms or [])}
    def match_map(p: Path) -> bool:
        if not sel:
            return True
        n = p.name.lower()
        return any(s in n for s in sel)
    imgs = [p for p in imgs if match_map(p)]
    # Gate-level circuits (encoding only)
    gates = [p for p in imgs if p.name.endswith("_gate_level.png") and (p.name.startswith("enhanced_") or p.name.startswith("preview_"))]
    # Quantum results only
    quantum = [p for p in imgs if (
        p.name.startswith("enhanced_") and (
            p.name.endswith("_actual_vs_predicted.png") or
            p.name.endswith("_roc_curve.png") or
            p.name.endswith("_pr_curve.png") or
            p.name.endswith("_confusion_matrix.png")
        )
    )]

    shown = False
    if gates:
        st.subheader("Gate-Level Circuits (encoding)")
        for p in sorted(gates):
            st.image(str(p), caption=p.name, width='stretch')
        shown = True
    if quantum:
        st.subheader("Quantum Results")
        for p in sorted(quantum):
            st.image(str(p), caption=p.name, width='stretch')
        shown = True
    if not shown:
        st.info("No quantum visualizations found yet.")

st.write("\n")
st.caption("Tip: Deploy to Streamlit Cloud and set IBM_API_KEY as a secret for live hardware kernels.")
