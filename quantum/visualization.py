"""
Visualization tools for quantum circuits, kernel matrices, and results.

This module provides functions to visualize quantum circuits, kernel matrices,
and results from quantum models.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.visualization import circuit_drawer, plot_histogram
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
from sklearn.model_selection import learning_curve


def setup_visualization(style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple[int, int] = (10, 8)):
    """
    Set up visualization style and defaults.
    
    Args:
        style: Matplotlib style to use
        figsize: Default figure size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    sns.set_context("notebook", font_scale=1.2)


def save_figure(fig, filename: str, results_dir: str = 'results', dpi: int = 300):
    """
    Save figure to results directory.
    
    Args:
        fig: Matplotlib figure to save
        filename: Filename to save the figure as
        results_dir: Directory to save the figure in
        dpi: DPI for the saved figure
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)


def _save_text_as_image(text: str, filename: str, results_dir: str = 'results', fontsize: int = 8):
    """
    Render plain text (monospace) to a PNG image for readability.

    If filename ends with .txt, it will be converted to .png.
    """
    import matplotlib.pyplot as _plt

    os.makedirs(results_dir, exist_ok=True)
    if filename.lower().endswith('.txt'):
        filename = filename[:-4] + '.png'
    filepath = os.path.join(results_dir, filename)

    lines = text.splitlines() if isinstance(text, str) else [str(text)]
    # Estimate figure size based on content
    max_len = max((len(line) for line in lines), default=80)
    width = max(6, min(20, max_len * 0.08))
    height = max(3, min(30, len(lines) * 0.3))

    fig, ax = _plt.subplots(figsize=(width, height))
    ax.axis('off')
    ax.text(0.01, 0.99, "\n".join(lines), family='monospace', fontsize=fontsize,
            va='top', ha='left', wrap=False)
    fig.tight_layout(pad=0.5)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    _plt.close(fig)
    print(f"Saved: {filepath}")


def visualize_circuit(
    circuit: Union[QuantumCircuit, ZZFeatureMap, PauliFeatureMap],
    filename: Optional[str] = None,
    output_format: str = 'mpl',
    results_dir: str = 'results',
    style: Optional[str] = 'default',
    with_gates: bool = True,
    reverse_bits: bool = False
):
    """
    Visualize a quantum circuit with enhanced details.
    
    Args:
        circuit: Quantum circuit to visualize
        filename: Filename to save the visualization as
        output_format: Output format for visualization ('mpl', 'text', 'latex')
        results_dir: Directory to save the visualization in
        style: Visualization style ('default', 'bw', 'paulis', 'gates', 'iqx', 'iqx-dark')
        with_gates: Show detailed gate information
        reverse_bits: Whether to reverse the bit order in the visualization
    """
    try:
        if output_format == 'mpl':
            # Use more detailed visualization options
            try:
                fig = circuit_drawer(
                    circuit,
                    output='mpl',
                    style=style,
                    plot_barriers=True,
                    fold=20,  # Fold wide circuits
                    idle_wires=False,  # Don't show unused wires
                    reverse_bits=reverse_bits,
                    scale=0.8,  # Adjust scale for better viewing
                    initial_state=True,  # Show initial state
                    cregbundle=False  # Don't bundle classical registers
                )
                if filename:
                    # ensure png extension for mpl output
                    if filename.lower().endswith('.txt'):
                        filename = filename[:-4] + '.png'
                    save_figure(fig, filename, results_dir=results_dir)
                return fig
            except Exception as e:
                # Fallback: render text drawing to an image for readability
                print(f"Warning: MPL circuit drawing failed: {e}")
                text_circuit = circuit_drawer(
                    circuit,
                    output='text',
                    fold=20,
                    reverse_bits=reverse_bits,
                    plot_barriers=True,
                    initial_state=True
                )
                # Convert TextDrawing to str
                if hasattr(text_circuit, '__str__'):
                    text_circuit = str(text_circuit)
                if with_gates and hasattr(circuit, 'data'):
                    gate_details = ["", "Detailed Gate Information:"]
                    for i, instruction in enumerate(circuit.data):
                        gate_name = instruction[0].name
                        # Robust qubit indexing across Qiskit versions
                        qubits = []
                        for q in instruction[1]:
                            try:
                                loc = circuit.find_bit(q)
                                idx = getattr(loc, 'index', None)
                                if idx is None:
                                    idx = circuit.qubits.index(q)
                                qubits.append(idx)
                            except Exception:
                                qubits.append(str(q))
                        params = []
                        if hasattr(instruction[0], 'params') and instruction[0].params:
                            params = [f"{p:.4f}" for p in instruction[0].params]
                        gate_info = f"Gate {i+1}: {gate_name} on qubit(s) {qubits}"
                        if params:
                            gate_info += f" with parameters {params}"
                        gate_details.append(gate_info)
                    text_circuit += "\n" + "\n".join(gate_details)
                if filename:
                    _save_text_as_image(text_circuit, filename, results_dir=results_dir)
                return text_circuit
        
        elif output_format == 'latex':
            # LaTeX output for professional publication-quality visualizations
            latex_circuit = circuit_drawer(
                circuit,
                output='latex',
                style=style,
                plot_barriers=True,
                reverse_bits=reverse_bits
            )
            
            if filename:
                os.makedirs(results_dir, exist_ok=True)
                filepath = os.path.join(results_dir, filename.replace('.txt', '.tex'))
                with open(filepath, 'w') as f:
                    f.write(latex_circuit)
                print(f"Saved LaTeX circuit to: {filepath}")
            return latex_circuit
        
        else:  # text
            # Enhanced text visualization with gate details -> render to PNG image for readability
            text_circuit = circuit_drawer(
                circuit,
                output='text',
                fold=20,  # Fold wide circuits
                reverse_bits=reverse_bits,
                plot_barriers=True,
                initial_state=True
            )
            if hasattr(text_circuit, '__str__'):
                text_circuit = str(text_circuit)
            gate_details = []
            if with_gates and hasattr(circuit, 'data'):
                gate_details.append("\nDetailed Gate Information:")
                for i, instruction in enumerate(circuit.data):
                    gate_name = instruction[0].name
                    qubits = []
                    for q in instruction[1]:
                        try:
                            loc = circuit.find_bit(q)
                            idx = getattr(loc, 'index', None)
                            if idx is None:
                                idx = circuit.qubits.index(q)
                            qubits.append(idx)
                        except Exception:
                            qubits.append(str(q))
                    params = []
                    if hasattr(instruction[0], 'params') and instruction[0].params:
                        params = [f"{p:.4f}" for p in instruction[0].params]
                    gate_info = f"Gate {i+1}: {gate_name} on qubit(s) {qubits}"
                    if params:
                        gate_info += f" with parameters {params}"
                    gate_details.append(gate_info)
                text_circuit += "\n" + "\n".join(gate_details)
            if filename:
                _save_text_as_image(text_circuit, filename, results_dir=results_dir)
            return text_circuit
    except Exception as e:
        error_msg = f"Warning: Could not visualize circuit: {e}"
        print(error_msg)
        
        # Create a simplified circuit representation as readable PNG fallback
        if filename:
            info_lines = [
                f"Circuit visualization failed: {e}",
                "",
            ]
            try:
                if hasattr(circuit, 'num_qubits'):
                    info_lines.append(f"Number of qubits: {circuit.num_qubits}")
                if hasattr(circuit, 'size'):
                    info_lines.append(f"Circuit size: {circuit.size()}")
                if hasattr(circuit, 'depth'):
                    info_lines.append(f"Circuit depth: {circuit.depth()}")
                if hasattr(circuit, 'count_ops'):
                    info_lines.append(f"Gate counts: {circuit.count_ops()}")
            except Exception:
                pass
            text_blob = "\n".join(info_lines)
            _save_text_as_image(text_blob, filename, results_dir=results_dir)
        
        return error_msg


def visualize_kernel_matrix(
    kernel_matrix: np.ndarray,
    title: str = "Kernel Matrix",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize a kernel matrix as a heatmap.
    
    Args:
        kernel_matrix: Kernel matrix to visualize
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(kernel_matrix, cmap="viridis", ax=ax)
    ax.set_title(title)
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_gate_level_circuit(
    circuit: QuantumCircuit,
    filename_prefix: str,
    results_dir: str = "results",
    backend: Any = None,
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 3,
    style: Optional[str] = "iqx",
):
    """
    Transpile and visualize a circuit at gate level (IBM-style basis).

    Args:
        circuit: Circuit to transpile and visualize.
        filename_prefix: Prefix for output files (without extension).
        results_dir: Directory to save visualizations.
        backend: Optional backend to target (e.g., IBM backend from Runtime).
        basis_gates: Optional explicit basis gates list (e.g., ['rz','sx','x','cx']).
        optimization_level: Transpile optimization level (0-3).
        style: Visualization style for mpl output.

    Produces PNG (mpl) and TXT (text) outputs.
    """
    os.makedirs(results_dir, exist_ok=True)

    try:
        if backend is not None:
            tc = transpile(circuit, backend=backend, optimization_level=optimization_level)
        else:
            # Default to IBM native basis if not provided
            if basis_gates is None:
                basis_gates = ["rz", "sx", "x", "cx"]
            tc = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    except Exception as e:
        print(f"Warning: transpile failed, falling back to original circuit: {e}")
        tc = circuit

    # Save MPL image (best effort); fallback to text-as-image
    img_name = f"{filename_prefix}_gate_level.png"
    try:
        fig = circuit_drawer(tc, output="mpl", style=style, fold=20)
        save_figure(fig, img_name, results_dir)
    except Exception as e:
        print(f"Warning: MPL circuit drawing failed: {e}")
        # Fallback to text drawing rendered as image
        text = circuit_drawer(tc, output="text", fold=20)
        _save_text_as_image(str(text), img_name, results_dir)
    # Also save raw text for reference
    try:
        txt_name = f"{filename_prefix}_gate_level.txt"
        path_txt = os.path.join(results_dir, txt_name)
        with open(path_txt, "w") as f:
            f.write(str(circuit_drawer(tc, output="text", fold=20)))
        print(f"Saved: {path_txt}")
    except Exception:
        pass
    return tc


def visualize_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize a ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {np.trapz(tpr, fpr):.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize a precision-recall curve for binary classification.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'AP = {np.mean(precision):.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize a confusion matrix for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize actual vs predicted values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs Predicted",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Plot residuals (y_true - y_pred) vs predicted values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title(title)

    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Histogram",
    filename: Optional[str] = None,
    results_dir: str = 'results',
    bins: int = 30
):
    """
    Plot histogram of residuals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title(title)

    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_residuals_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Q-Q Plot",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Q-Q plot of residuals against a normal distribution.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)

    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_feature_distribution(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Feature Distribution",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize the distribution of features colored by class.
    
    Args:
        X: Features
        y: Labels
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    # Ensure numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Print debug info
    print(f"X shape in visualize_feature_distribution: {X.shape}")
    
    # Handle different input shapes
    if X.ndim == 0:
        # Handle scalar input
        X = np.array([[X]])
    elif X.ndim == 1:
        # Handle 1D array - convert to 2D column vector
        X = X.reshape(-1, 1)
        
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if X.ndim < 2:
        # Fallback for unusual cases
        ax.text(0.5, 0.5, f"Cannot visualize data with shape {X.shape}",
                ha='center', va='center', fontsize=12)
        ax.set_title(f"{title} (Visualization not possible)")
    elif X.shape[1] == 1:
        # If 1 feature, plot as histogram
        unique_y = np.unique(y)
        for label in unique_y:
            subset = X[y == label].flatten()
            ax.hist(subset, alpha=0.5, bins=20, label=f'Class {label}')
        ax.legend()
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Count')
        ax.set_title(f"{title} (1D)")
    elif X.shape[1] == 2:
        # If 2 features, plot directly
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
        fig.colorbar(scatter, label='Class/Value')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
    else:
        # If more than 2 features, use PCA to reduce dimensionality
        try:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
            fig.colorbar(scatter, label='Class/Value')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(f"{title} (PCA)")
        except Exception as e:
            # Fallback if PCA fails
            print(f"PCA failed: {e}")
            ax.text(0.5, 0.5, f"PCA failed: {e}", ha='center', va='center', fontsize=12)
            ax.set_title(f"{title} (Visualization failed)")
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_histogram(
    counts: Dict[str, int],
    title: str = "Measurement Counts",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize measurement counts as a histogram.
    
    Args:
        counts: Dictionary of measurement counts
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    fig = plot_histogram(counts, title=title)
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "Model Comparison",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Visualize a comparison of metrics across different models.
    
    Args:
        metrics_dict: Dictionary of metrics for different models
        metric_name: Name of the metric to compare
        title: Title for the visualization
        filename: Filename to save the visualization as
        results_dir: Directory to save the visualization in
    """
    models = list(metrics_dict.keys())
    values = [metrics_dict[model]["metrics"][metric_name] for model in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name)
    ax.set_title(f"{title}: {metric_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    
    return fig


def visualize_angle_distributions(
    X_angles: np.ndarray,
    y: Optional[np.ndarray] = None,
    title: str = "Encoded Angle Distributions",
    filename: Optional[str] = None,
    results_dir: str = 'results',
    max_features: int = 8,
    bins: int = 30
):
    """
    Plot histograms of encoded rotation angles per dimension. If more than
    max_features, only the first max_features are shown.
    """
    X = np.asarray(X_angles)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_feats = X.shape[1]
    k = min(max_features, n_feats)

    cols = min(4, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(k):
        ax = axes[i]
        if y is None:
            ax.hist(X[:, i], bins=bins, color='slateblue', alpha=0.8)
        else:
            y_arr = np.asarray(y)
            if len(np.unique(y_arr)) <= 5:
                for cls in np.unique(y_arr):
                    ax.hist(X[y_arr == cls, i], bins=bins, alpha=0.5, label=f"{cls}")
                ax.legend(fontsize=8)
            else:
                q = np.quantile(y_arr, [0.33, 0.66])
                mask_low = y_arr <= q[0]
                mask_mid = (y_arr > q[0]) & (y_arr <= q[1])
                mask_high = y_arr > q[1]
                ax.hist(X[mask_low, i], bins=bins, alpha=0.5, label="low")
                ax.hist(X[mask_mid, i], bins=bins, alpha=0.5, label="mid")
                ax.hist(X[mask_high, i], bins=bins, alpha=0.5, label="high")
                ax.legend(fontsize=8)
        ax.set_title(f"Angle {i}")
    for j in range(k, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(title)
    fig.tight_layout()

    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_correlation_matrix(
    X: np.ndarray,
    title: str = "Feature Correlation",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Heatmap of Pearson correlation between features.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title(title)
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_pca_variance(
    X: np.ndarray,
    title: str = "PCA Explained Variance",
    filename: Optional[str] = None,
    results_dir: str = 'results'
):
    """
    Scree plot of PCA explained variance ratios.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_components = min( min(X.shape[0], X.shape[1]), 20)
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components)
    try:
        pca.fit(X)
        evr = pca.explained_variance_ratio_
    except Exception:
        evr = np.array([1.0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(evr)+1), evr, marker='o')
    ax.set_xlabel('Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Learning Curve",
    filename: Optional[str] = None,
    results_dir: str = 'results',
    cv: int = 3,
    scoring: Optional[str] = None,
    train_sizes: Optional[np.ndarray] = None,
    n_jobs: int = -1
):
    """
    Plot training and cross-validation scores vs. training set size.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)

    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes,
        shuffle=True, random_state=42
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, np.mean(train_scores, axis=1), 'o-', label='Train')
    ax.plot(sizes, np.mean(val_scores, axis=1), 'o-', label='CV')
    ax.fill_between(sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.2)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score' if scoring is None else scoring)
    ax.set_title(title)
    ax.legend()

    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig


def visualize_learning_curve_points(
    sizes: np.ndarray,
    train_scores: Optional[np.ndarray],
    val_scores: np.ndarray,
    title: str = "Learning Curve",
    filename: Optional[str] = None,
    results_dir: str = 'results',
    ylabel: str = 'Score'
):
    """
    Plot a learning-curve-like chart from precomputed scores.
    """
    sizes = np.asarray(sizes)
    val_scores = np.asarray(val_scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_scores is not None:
        ax.plot(sizes, np.asarray(train_scores), 'o-', label='Train')
    ax.plot(sizes, val_scores, 'o-', label='Validation')
    ax.set_xlabel('Training examples')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if filename:
        save_figure(fig, filename, results_dir=results_dir)
    return fig
