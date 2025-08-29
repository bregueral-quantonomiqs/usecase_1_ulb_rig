# Quantum Insurance – User Guide

This guide shows how to set up the environment from scratch and run the pipelines (classical, quantum, and Streamlit demo). It assumes macOS/Linux and Python 3.11+ (3.12 recommended).

## 1) Clone and set up environment

- Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

Verify versions (optional):

```bash
python -c "import sklearn, qiskit; print('sklearn', sklearn.__version__); print('qiskit', qiskit.__version__)"
```

## 2) IBM Quantum credentials (optional, for hardware/backends)

You can provide your IBM API key via either:

- Environment variable:

```bash
export IBM_API_KEY="<your_token>"
```

- Or a config file `config/ibm_config.json`:

```json
{ "IBM_API_KEY": "<your_token>" }
```

If a token is set, the quantum kernel code will try to use IBM Runtime when you pass an IBM backend name.

## 3) Run the pipelines

All outputs are written under `results/` (plots and JSON) and are ignored by git (the folder is kept with a `.gitkeep`).

### A. End‑to‑end (occurrence + severity)

```bash
python -m quantum.run_end_to_end
```

or with the CLI for more control:

```bash
python quantum/enhanced_main.py \
  --data-path data/mcc.csv \
  --qubits 8 \
  --backend qasm_simulator \
  --results-dir results \
  --skip-quantum-lc         # optional, faster
  --skip-svm-calibration    # optional, faster
```

### B. Severity only (classical baseline, quick)

```bash
python quantum/run_severity_analysis.py
```

### C. Severity only (quantum kernel via Qiskit)

Hardware‑friendly sizes and a reasonable map:

```bash
python quantum/run_severity_qiskit.py \
  --data-path data/mcc.csv \
  --qubits 6 \
  --backend ibm_qasm_simulator \
  --n-train 40 \
  --n-test 20 \
  --results-dir results \
  --feature-maps zz_circular
```

To run on hardware, change backend to your device name (e.g., `--backend ibm_tokyo`) and ensure your IBM token is configured:

```bash
python quantum/run_severity_qiskit.py --backend ibm_tokyo --qubits 5 --n-train 20 --n-test 10 --feature-maps zz_circular
```

## 4) Streamlit demo

Start the web demo locally:

```bash
streamlit run app/streamlit_app.py
```

In the sidebar, choose your dataset, number of qubits, and backend. Click “Run Pipeline” to execute the workflow and view metrics/plots. If an IBM token is detected and you set an IBM backend name, the app attempts to use IBM Runtime automatically.

## 5) Cleaning outputs

To clear generated outputs and caches (keeps `.venv` and sources):

```bash
make clean
# or
bash scripts/clean.sh
```

## 6) Notes

- Circuit visualizations prefer Matplotlib; when not available, they are rendered from text into PNG for readability.
- Quantum kernel runs scale poorly with dataset size; use small `--n-train/--n-test` values for hardware demos.
- Results are saved under `results/`; the directory is kept empty in git via `.gitkeep`.

