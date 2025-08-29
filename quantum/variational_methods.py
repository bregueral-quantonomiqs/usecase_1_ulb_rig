"""
Variational quantum methods for classification and regression.

This module provides functions to create and train variational quantum
classifiers (VQC) and variational quantum regressors (VQR).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, RealAmplitudes, TwoLocal
# algorithm_globals is no longer available in qiskit.utils in Qiskit 2.x
# Using numpy's random seed directly instead
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def create_variational_circuit(
    feature_dimension: int = 8,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    feature_map_type: str = 'zz',
    entanglement: str = 'full',
    rotation_blocks: List[str] = None
) -> QuantumCircuit:
    """
    Create a variational quantum circuit for VQC/VQR.
    
    Args:
        feature_dimension: Number of qubits/features
        feature_map_reps: Number of repetitions of the feature map
        ansatz_reps: Number of repetitions of the ansatz
        feature_map_type: Type of feature map ('zz' or 'pauli')
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        rotation_blocks: List of rotation gates to use in the ansatz
        
    Returns:
        QuantumCircuit: A variational quantum circuit
    """
    # Create feature map
    if feature_map_type.lower() == 'zz':
        feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=feature_map_reps,
            entanglement=entanglement
        )
    else:  # pauli
        feature_map = PauliFeatureMap(
            feature_dimension=feature_dimension,
            reps=feature_map_reps,
            entanglement=entanglement,
            paulis=['Z', 'ZZ']
        )
    
    # Create ansatz
    if rotation_blocks is None:
        rotation_blocks = ['ry', 'rz']
    
    ansatz = TwoLocal(
        feature_dimension,
        rotation_blocks=rotation_blocks,
        entanglement_blocks='cz',
        reps=ansatz_reps,
        entanglement=entanglement
    )
    
    # Combine feature map and ansatz
    circuit = QuantumCircuit(feature_dimension)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    
    return circuit


def create_parameterized_circuit(
    feature_dimension: int = 8,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    feature_map_type: str = 'zz',
    entanglement: str = 'full'
) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """
    Create a parameterized quantum circuit for VQC/VQR.
    
    Args:
        feature_dimension: Number of qubits/features
        feature_map_reps: Number of repetitions of the feature map
        ansatz_reps: Number of repetitions of the ansatz
        feature_map_type: Type of feature map ('zz' or 'pauli')
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        
    Returns:
        Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
            - Parameterized quantum circuit
            - List of data parameters
            - List of trainable parameters
    """
    # Create feature map
    if feature_map_type.lower() == 'zz':
        feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=feature_map_reps,
            entanglement=entanglement
        )
    else:  # pauli
        feature_map = PauliFeatureMap(
            feature_dimension=feature_dimension,
            reps=feature_map_reps,
            entanglement=entanglement,
            paulis=['Z', 'ZZ']
        )
    
    # Create ansatz
    ansatz = TwoLocal(
        feature_dimension,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cz',
        reps=ansatz_reps,
        entanglement=entanglement
    )
    
    # Combine feature map and ansatz
    circuit = QuantumCircuit(feature_dimension)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    
    # Add measurement
    circuit.measure_all()
    
    # Get data and trainable parameters
    data_params = feature_map.parameters
    trainable_params = ansatz.parameters
    
    return circuit, list(data_params), list(trainable_params)


def execute_circuit(
    circuit: QuantumCircuit,
    params_dict: Dict[Parameter, float],
    backend_name: str = 'statevector_simulator',
    shots: int = 1024
) -> Dict[str, int]:
    """
    Execute a quantum circuit with the given parameters.
    
    Args:
        circuit: Quantum circuit to execute
        params_dict: Dictionary of parameters to bind
        backend_name: Name of the backend to use for simulation
        shots: Number of shots for the simulation
        
    Returns:
        Dict[str, int]: Measurement counts
    """
    backend = Aer.get_backend(backend_name)
    bound_circuit = circuit.assign_parameters(params_dict)
    transpiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    return job.result().get_counts()


def binary_classification_objective(
    theta: np.ndarray,
    circuit: QuantumCircuit,
    data_params: List[Parameter],
    trainable_params: List[Parameter],
    X_train: np.ndarray,
    y_train: np.ndarray,
    backend_name: str = 'statevector_simulator',
    shots: int = 1024
) -> float:
    """
    Objective function for binary classification.
    
    Args:
        theta: Trainable parameters
        circuit: Quantum circuit
        data_params: Data parameters
        trainable_params: Trainable parameters
        X_train: Training data
        y_train: Training labels
        backend_name: Name of the backend to use for simulation
        shots: Number of shots for the simulation
        
    Returns:
        float: Loss value
    """
    loss = 0.0
    
    for i, x in enumerate(X_train):
        # Bind data and trainable parameters
        params_dict = {}
        for j, param in enumerate(data_params):
            params_dict[param] = x[j % len(x)]  # Cycle through features if needed
        for j, param in enumerate(trainable_params):
            params_dict[param] = theta[j]
        
        # Execute circuit
        counts = execute_circuit(circuit, params_dict, backend_name, shots)
        
        # Compute prediction
        if '0' in counts:
            p0 = counts['0'] / shots
        else:
            p0 = 0.0
        
        # Binary cross-entropy loss
        y = y_train[i]
        if y == 0:
            loss -= np.log(p0 + 1e-10)
        else:
            loss -= np.log(1 - p0 + 1e-10)
    
    return loss / len(X_train)


def regression_objective(
    theta: np.ndarray,
    circuit: QuantumCircuit,
    data_params: List[Parameter],
    trainable_params: List[Parameter],
    X_train: np.ndarray,
    y_train: np.ndarray,
    backend_name: str = 'statevector_simulator',
    shots: int = 1024
) -> float:
    """
    Objective function for regression.
    
    Args:
        theta: Trainable parameters
        circuit: Quantum circuit
        data_params: Data parameters
        trainable_params: Trainable parameters
        X_train: Training data
        y_train: Training labels
        backend_name: Name of the backend to use for simulation
        shots: Number of shots for the simulation
        
    Returns:
        float: Loss value
    """
    loss = 0.0
    
    for i, x in enumerate(X_train):
        # Bind data and trainable parameters
        params_dict = {}
        for j, param in enumerate(data_params):
            params_dict[param] = x[j % len(x)]  # Cycle through features if needed
        for j, param in enumerate(trainable_params):
            params_dict[param] = theta[j]
        
        # Execute circuit
        counts = execute_circuit(circuit, params_dict, backend_name, shots)
        
        # Compute prediction (expectation value)
        expectation = 0.0
        for bitstring, count in counts.items():
            # Convert bitstring to value
            value = 0.0
            for j, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    value += 2**j
            expectation += value * count / shots
        
        # Normalize to [0, 1]
        n_qubits = circuit.num_qubits
        expectation /= (2**n_qubits - 1)
        
        # Scale to match target range
        y_min, y_max = np.min(y_train), np.max(y_train)
        prediction = expectation * (y_max - y_min) + y_min
        
        # Mean squared error
        loss += (prediction - y_train[i])**2
    
    return loss / len(X_train)


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dimension: int = 8,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    feature_map_type: str = 'zz',
    entanglement: str = 'full',
    backend_name: str = 'statevector_simulator',
    shots: int = 1024,
    random_seed: int = 42
) -> Dict:
    """
    Train a Variational Quantum Classifier.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        feature_dimension: Number of qubits/features
        feature_map_reps: Number of repetitions of the feature map
        ansatz_reps: Number of repetitions of the ansatz
        feature_map_type: Type of feature map ('zz' or 'pauli')
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        backend_name: Name of the backend to use for simulation
        shots: Number of shots for the simulation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict: Dictionary of training results
    """
    # Set random seed
    # algorithm_globals.random_seed is no longer available in Qiskit 2.x
    np.random.seed(random_seed)
    
    # Create parameterized circuit
    circuit, data_params, trainable_params = create_parameterized_circuit(
        feature_dimension=feature_dimension,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        feature_map_type=feature_map_type,
        entanglement=entanglement
    )
    
    # Initialize trainable parameters
    initial_theta = np.random.uniform(0, 2*np.pi, len(trainable_params))
    
    # Define objective function
    def objective(theta):
        return binary_classification_objective(
            theta, circuit, data_params, trainable_params,
            X_train, y_train, backend_name, shots
        )
    
    # Optimize
    result = minimize(
        objective,
        initial_theta,
        method='COBYLA',
        options={'maxiter': 100}
    )
    
    # Get optimal parameters
    optimal_theta = result.x
    
    # Predict
    y_pred = []
    y_score = []
    
    for x in X_test:
        # Bind data and trainable parameters
        params_dict = {}
        for j, param in enumerate(data_params):
            params_dict[param] = x[j % len(x)]  # Cycle through features if needed
        for j, param in enumerate(trainable_params):
            params_dict[param] = optimal_theta[j]
        
        # Execute circuit
        counts = execute_circuit(circuit, params_dict, backend_name, shots)
        
        # Compute prediction
        if '0' in counts:
            p0 = counts['0'] / shots
        else:
            p0 = 0.0
        
        y_pred.append(0 if p0 > 0.5 else 1)
        y_score.append(1 - p0)  # Probability of class 1
    
    # Evaluate
    from .kernel_methods import evaluate_classification
    metrics = evaluate_classification(y_test, np.array(y_pred), np.array(y_score))
    
    return {
        "metrics": metrics,
        "y_pred": np.array(y_pred),
        "y_score": np.array(y_score),
        "optimal_params": optimal_theta
    }


def train_vqr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dimension: int = 8,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    feature_map_type: str = 'zz',
    entanglement: str = 'full',
    backend_name: str = 'statevector_simulator',
    shots: int = 1024,
    random_seed: int = 42
) -> Dict:
    """
    Train a Variational Quantum Regressor.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        feature_dimension: Number of qubits/features
        feature_map_reps: Number of repetitions of the feature map
        ansatz_reps: Number of repetitions of the ansatz
        feature_map_type: Type of feature map ('zz' or 'pauli')
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        backend_name: Name of the backend to use for simulation
        shots: Number of shots for the simulation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict: Dictionary of training results
    """
    # Set random seed
    # algorithm_globals.random_seed is no longer available in Qiskit 2.x
    np.random.seed(random_seed)
    
    # Create parameterized circuit
    circuit, data_params, trainable_params = create_parameterized_circuit(
        feature_dimension=feature_dimension,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        feature_map_type=feature_map_type,
        entanglement=entanglement
    )
    
    # Initialize trainable parameters
    initial_theta = np.random.uniform(0, 2*np.pi, len(trainable_params))
    
    # Define objective function
    def objective(theta):
        return regression_objective(
            theta, circuit, data_params, trainable_params,
            X_train, y_train, backend_name, shots
        )
    
    # Optimize
    result = minimize(
        objective,
        initial_theta,
        method='COBYLA',
        options={'maxiter': 100}
    )
    
    # Get optimal parameters
    optimal_theta = result.x
    
    # Predict
    y_pred = []
    
    for x in X_test:
        # Bind data and trainable parameters
        params_dict = {}
        for j, param in enumerate(data_params):
            params_dict[param] = x[j % len(x)]  # Cycle through features if needed
        for j, param in enumerate(trainable_params):
            params_dict[param] = optimal_theta[j]
        
        # Execute circuit
        counts = execute_circuit(circuit, params_dict, backend_name, shots)
        
        # Compute prediction (expectation value)
        expectation = 0.0
        for bitstring, count in counts.items():
            # Convert bitstring to value
            value = 0.0
            for j, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    value += 2**j
            expectation += value * count / shots
        
        # Normalize to [0, 1]
        n_qubits = circuit.num_qubits
        expectation /= (2**n_qubits - 1)
        
        # Scale to match target range
        y_min, y_max = np.min(y_train), np.max(y_train)
        prediction = expectation * (y_max - y_min) + y_min
        
        y_pred.append(prediction)
    
    # Evaluate
    from .kernel_methods import evaluate_regression
    metrics = evaluate_regression(y_test, np.array(y_pred))
    
    return {
        "metrics": metrics,
        "y_pred": np.array(y_pred),
        "optimal_params": optimal_theta
    }


def run_variational_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dimension: int = 8,
    feature_map_types: List[str] = None,
    entanglement_patterns: List[str] = None,
    backend_name: str = 'statevector_simulator'
) -> Dict:
    """
    Run variational quantum classification with multiple configurations.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        feature_dimension: Number of qubits/features
        feature_map_types: List of feature map types to use
        entanglement_patterns: List of entanglement patterns to use
        backend_name: Name of the backend to use for simulation
        
    Returns:
        Dict: Dictionary of results for each configuration
    """
    if feature_map_types is None:
        feature_map_types = ['zz', 'pauli']
    
    if entanglement_patterns is None:
        entanglement_patterns = ['linear', 'circular', 'full']
    
    results = {}
    
    for fm_type in feature_map_types:
        for pattern in entanglement_patterns:
            name = f"vqc_{fm_type}_{pattern}"
            print(f"Running variational quantum classification with {name}...")
            
            # Train and evaluate
            result = train_vqc(
                X_train, y_train, X_test, y_test,
                feature_dimension=feature_dimension,
                feature_map_type=fm_type,
                entanglement=pattern,
                backend_name=backend_name
            )
            
            # Store results
            results[name] = result
    
    return results


def run_variational_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dimension: int = 8,
    feature_map_types: List[str] = None,
    entanglement_patterns: List[str] = None,
    backend_name: str = 'statevector_simulator'
) -> Dict:
    """
    Run variational quantum regression with multiple configurations.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        feature_dimension: Number of qubits/features
        feature_map_types: List of feature map types to use
        entanglement_patterns: List of entanglement patterns to use
        backend_name: Name of the backend to use for simulation
        
    Returns:
        Dict: Dictionary of results for each configuration
    """
    if feature_map_types is None:
        feature_map_types = ['zz', 'pauli']
    
    if entanglement_patterns is None:
        entanglement_patterns = ['linear', 'circular', 'full']
    
    results = {}
    
    for fm_type in feature_map_types:
        for pattern in entanglement_patterns:
            name = f"vqr_{fm_type}_{pattern}"
            print(f"Running variational quantum regression with {name}...")
            
            # Train and evaluate
            result = train_vqr(
                X_train, y_train, X_test, y_test,
                feature_dimension=feature_dimension,
                feature_map_type=fm_type,
                entanglement=pattern,
                backend_name=backend_name
            )
            
            # Store results
            results[name] = result
    
    return results