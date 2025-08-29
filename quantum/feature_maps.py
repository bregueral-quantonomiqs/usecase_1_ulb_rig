"""
Feature maps for quantum encoding of classical data.

This module provides functions to create different types of feature maps
with various entanglement patterns for quantum encoding of classical data.
"""

from typing import List, Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.visualization import circuit_drawer


def create_zz_feature_map(
    feature_dimension: int = 8,
    reps: int = 2,
    entanglement: str = 'linear',
    name: str = 'ZZFeatureMap'
) -> ZZFeatureMap:
    """
    Create a ZZFeatureMap with specified entanglement pattern.
    
    Args:
        feature_dimension: Number of qubits/features
        reps: Number of repetitions of the feature map
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        name: Name of the feature map
        
    Returns:
        ZZFeatureMap: A ZZFeatureMap with the specified parameters
    """
    return ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
        name=name
    )


def create_pauli_feature_map(
    feature_dimension: int = 8,
    reps: int = 2,
    entanglement: str = 'linear',
    paulis: Optional[List[str]] = None,
    name: str = 'PauliFeatureMap'
) -> PauliFeatureMap:
    """
    Create a PauliFeatureMap with specified entanglement pattern.
    
    Args:
        feature_dimension: Number of qubits/features
        reps: Number of repetitions of the feature map
        entanglement: Entanglement strategy ('linear', 'circular', 'full')
        paulis: List of Pauli strings to use in the feature map
        name: Name of the feature map
        
    Returns:
        PauliFeatureMap: A PauliFeatureMap with the specified parameters
    """
    if paulis is None:
        paulis = ['Z', 'ZZ']
        
    return PauliFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
        paulis=paulis,
        name=name
    )


def visualize_feature_map(
    feature_map: Union[ZZFeatureMap, PauliFeatureMap],
    sample_input: Optional[np.ndarray] = None,
    output_format: str = 'mpl'
) -> Union[QuantumCircuit, str]:
    """
    Visualize a feature map circuit.
    
    Args:
        feature_map: The feature map to visualize
        sample_input: Sample input data to bind to the feature map parameters
        output_format: Output format for visualization ('mpl', 'text')
        
    Returns:
        Visualization of the feature map circuit
    """
    circuit = feature_map
    
    if sample_input is not None:
        # Bind parameters to create a concrete circuit
        bound_circuit = circuit.bind_parameters({
            circuit.parameters[i]: sample_input[i]
            for i in range(min(len(circuit.parameters), len(sample_input)))
        })
        circuit = bound_circuit
    
    return circuit_drawer(circuit, output=output_format)


def get_all_feature_maps(feature_dimension: int = 8, reps: int = 2) -> dict:
    """
    Create all combinations of feature maps and entanglement patterns.
    
    Args:
        feature_dimension: Number of qubits/features
        reps: Number of repetitions of the feature map
        
    Returns:
        dict: Dictionary of feature maps with different configurations
    """
    entanglement_patterns = ['linear', 'circular', 'full']
    feature_maps = {}
    
    # Create ZZFeatureMaps with different entanglement patterns
    for pattern in entanglement_patterns:
        name = f'zz_{pattern}'
        feature_maps[name] = create_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=pattern,
            name=name
        )
    
    # Create PauliFeatureMaps with different entanglement patterns
    for pattern in entanglement_patterns:
        name = f'pauli_{pattern}'
        feature_maps[name] = create_pauli_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=pattern,
            name=name
        )
    
    return feature_maps